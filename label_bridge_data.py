"""
Script to generate paths and masks for Bridge data using direct VLM inference.
This script should be run before convert_bridge_data_to_lerobot.py to prepare the path/mask data.

Creates an h5 file with the following structure:

episode_group
    .attrs:
        task_description: str
        num_steps: uint16
        has_paths: bool
        has_masks: bool
        cameras_with_data: list of strings - which cameras have valid data
        total_images: uint16 - total number of images across all cameras
    
    Per-camera datasets (only created if camera has valid images):
    {camera}_paths: 2D array of shape (num_images_for_camera, max_path_length, 2) - paths for this camera
    {camera}_path_lengths: 1D array of shape (num_images_for_camera) - length of each path for this camera
    {camera}_path_timesteps: 1D array of shape (num_images_for_camera) - 0-indexed step numbers for this camera's paths
    {camera}_masks: 2D array of shape (num_images_for_camera, max_mask_length, 2) - masks for this camera  
    {camera}_mask_lengths: 1D array of shape (num_images_for_camera) - length of each mask for this camera
    {camera}_mask_timesteps: 1D array of shape (num_images_for_camera) - 0-indexed step numbers for this camera's masks

    Where {camera} can be: image_0, image_1, image_2, image_3

Note: Each camera gets separate datasets, making it easy to access camera-specific paths/masks without filtering.

pip install tensorflow-datasets
pip install h5py
pip install -e ~/vila_utils
pip install shapely
pip install tensorflow


CUDA_VISIBLE_DEVICES=1 python label_bridge_data.py \
    --args.data-dir=/data/shared/openx_rlds_data/ \
    --args.output-dir=./test_bridge_labelin \
    --args.model-path ~/.cache/huggingface/hub/models--memmelma--vila_3b_path_mask/snapshots/943d8524c570c424043b17f9c623c9c37648cec3/checkpoint-6600/ \
    --args.batch-size=16
"""

import dataclasses
import logging
import pathlib
from pathlib import Path
import tensorflow_datasets as tfds
import numpy as np
import tqdm
import tyro
import h5py
import torch
import cv2
from PIL import Image
from typing import List, Tuple, Optional

PROMPT_TYPE = "path_mask"

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from vila_utils.utils.prompts import get_prompt
from vila_utils.utils.decode import (
    add_path_2d_to_img_alt_fast,
    add_mask_2d_to_img,
    get_path_from_answer,
)
from vila_utils.utils.encode import scale_path
CONV_MODE = "vicuna_v1"

@dataclasses.dataclass
class Args:
    data_dir: str  # Directory containing the Bridge dataset
    output_dir: str  # Directory to save the generated paths and masks
    model_path: str  # Path to the VLM model
    resize_size: int = 224 # Size to resize images for VLM
    draw_path: bool = True  # Whether to generate paths
    draw_mask: bool = True  # Whether to generate masks
    flip_image_horizontally: bool = False  # Whether to flip images horizontally
    batch_size: int = 1  # Batch size for inference (1 for single, >1 for batched)
    temperature: float = 0.1
    top_p: Optional[float] = 0.95
    max_new_tokens: int = 512
    num_beams: int = 1
    vlm_call_frequency: int = 20  # Save every N timesteps


def normalize_image_tags(model, qs: str) -> str:
    """Normalize image tags in the query string."""
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = qs.replace(IMAGE_PLACEHOLDER, image_token_se)
        else:
            qs = qs.replace(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN)
    return qs


def parse_vlm_output(
    output: str, stop_str: str = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Parse VLM output to extract path and mask coordinates.

    Args:
        output: Raw output from VLM
        stop_str: Stop string to remove from output

    Returns:
        Tuple of (path_coords, mask_coords)
    """
    # Clean up output
    output = output.strip()
    if stop_str and output.endswith(stop_str):
        output = output[: -len(stop_str)].strip()

    # Extract path and mask using vila_utils
    path, mask = get_path_from_answer(output, PROMPT_TYPE)

    return path, mask


def get_path_mask_from_vlm_direct(
    images: List[np.ndarray],
    task_descriptions: List[str],
    model,
    tokenizer,
    image_processor,
    args: Args,
    device,
) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    """
    Get path and mask predictions from VLM using direct inference.

    Args:
        images: List of image arrays
        task_descriptions: List of task descriptions
        model: Loaded VLM model
        tokenizer: Model tokenizer
        image_processor: Image processor
        args: Arguments containing inference parameters
        device: Device to run inference on

    Returns:
        Tuple of (paths, masks) lists
    """
    paths = []
    masks = []

    # Convert numpy images to PIL
    pil_images = []
    for img in images:
        if args.flip_image_horizontally:
            img = img[:, ::-1]
        # Convert to RGB if needed -- old code, not needed as we're loading directly from the tfds
        #if len(img.shape) == 3 and img.shape[2] == 3:
        #    rgb_img = (
        #        cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.dtype == np.uint8 else img
        #    )
        #else:
        rgb_img = img
        pil_image = Image.fromarray(rgb_img.astype(np.uint8))
        pil_images.append(pil_image)

    # Choose processing method based on batch_size
    if args.batch_size == 1:
        # Process each image individually (more reliable for debugging)
        for pil_img, task_desc in zip(pil_images, task_descriptions):
            failed = True
            while failed:
                try:
                    # Create conversation template (following vila_3b_server.py pattern)
                    conv = conv_templates[CONV_MODE].copy()
                    user_role = conv.roles[0]

                    # Create query for path and mask prediction
                    query = get_prompt(task_desc, PROMPT_TYPE, prompt_eval=True)
                    query = f"{IMAGE_PLACEHOLDER}{query}"

                    if query is None:
                        paths.append(None)
                        masks.append(None)
                        continue

                    # Normalize image tags
                    normalized_query = normalize_image_tags(model, query)

                    # Add messages to conversation
                    conv.append_message(user_role, normalized_query)

                    # Get the full prompt
                    prompt_text = conv.get_prompt()

                    # Process image
                    images_tensor = process_images(
                        [pil_img], image_processor, model.config
                    ).to(device, dtype=torch.float16)

                    # Tokenize prompt
                    input_ids = (
                        tokenizer_image_token(
                            prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                        )
                        .unsqueeze(0)
                        .to(device)
                    )

                    # Set up stopping criteria (following vila_3b_server.py pattern)
                    stop_str = (
                        conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    )
                    keywords = [stop_str] if stop_str else []
                    stopping_criteria = (
                        [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)]
                        if keywords
                        else []
                    )

                    # Generate response
                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=[images_tensor],
                            do_sample=True if args.temperature > 0 else False,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=args.num_beams,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=True,
                            stopping_criteria=stopping_criteria
                            if stopping_criteria
                            else None,
                        )

                    # Decode output
                    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                    path, mask = parse_vlm_output(output, stop_str)

                    paths.append(path)
                    masks.append(mask)
                    failed = False

                except Exception as e:
                    logging.error(f"Error during VLM inference: {e}")
                    paths.append(None)
                    masks.append(None)

    else:
        # Batched inference processing
        batch_size = min(args.batch_size, len(images))

        for i in range(0, len(images), batch_size):
            batch_images = pil_images[i : i + batch_size]
            batch_tasks = task_descriptions[i : i + batch_size]

            failed = True
            while failed:

                try:
                    # Prepare prompts for batch
                    prompts = []
                    valid_images = []
                    batch_results_map = []  # Track which images have valid queries

                    for idx, (pil_img, task_desc) in enumerate(
                        zip(batch_images, batch_tasks)
                    ):
                        # Create conversation template
                        conv = conv_templates[CONV_MODE].copy()
                        user_role = conv.roles[0]

                        # Create query for path and mask prediction
                        query = get_prompt(task_desc, PROMPT_TYPE, prompt_eval=True)
                        query = f"{IMAGE_PLACEHOLDER}{query}"

                        if query is None:
                            batch_results_map.append(None)  # Mark as skipped
                            continue

                        # Normalize image tags and build conversation
                        normalized_query = normalize_image_tags(model, query)
                        conv.append_message(user_role, normalized_query)

                        prompt_text = conv.get_prompt()
                        prompts.append(prompt_text)
                        valid_images.append(pil_img)
                        batch_results_map.append(
                            len(valid_images) - 1
                        )  # Map to index in results

                    if not prompts:
                        # Add None results for this batch
                        for _ in range(len(batch_images)):
                            paths.append(None)
                            masks.append(None)
                        continue

                    # Process images for batch
                    images_tensor = process_images(
                        valid_images, image_processor, model.config
                    ).to(device, dtype=torch.float16)

                    # Tokenize all prompts
                    batch_input_ids = [
                        tokenizer_image_token(
                            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                        )
                        .squeeze(0)
                        .to(device)
                        for prompt in prompts
                    ]

                    # Pad sequences to same length for batching
                    max_len = max([len(seq) for seq in batch_input_ids])
                    padded_input_ids = []
                    for seq in batch_input_ids:
                        if len(seq) >= max_len:
                            padded_seq = seq[:max_len]
                        else:
                            pad_token_id = (
                                tokenizer.pad_token_id
                                if tokenizer.pad_token_id is not None
                                else 0
                            )
                            padding = torch.full(
                                (max_len - len(seq),),
                                pad_token_id,
                                dtype=seq.dtype,
                                device=seq.device,
                            )
                            padded_seq = torch.cat([seq, padding])
                        padded_input_ids.append(padded_seq)

                    batch_input_ids = torch.stack(padded_input_ids)

                    # Set up stopping criteria
                    conv_temp = conv_templates[CONV_MODE].copy()
                    stop_str = (
                        conv_temp.sep
                        if conv_temp.sep_style != SeparatorStyle.TWO
                        else conv_temp.sep2
                    )
                    keywords = [stop_str] if stop_str else []
                    stopping_criteria = (
                        [KeywordsStoppingCriteria(keywords, tokenizer, batch_input_ids)]
                        if keywords
                        else []
                    )

                    # Generate responses for batch
                    with torch.inference_mode():
                        output_ids = model.generate(
                            batch_input_ids,
                            images=images_tensor,
                            do_sample=True if args.temperature > 0 else False,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=args.num_beams,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=True,
                            stopping_criteria=stopping_criteria
                            if stopping_criteria
                            else None,
                        )

                    # Decode outputs
                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                    # Process each output in the batch using the results map
                    for idx, result_idx in enumerate(batch_results_map):
                        if result_idx is None:
                            # This image was skipped (no valid query)
                            paths.append(None)
                            masks.append(None)
                        else:
                            # This image has a valid result
                            output = outputs[result_idx]
                            path, mask = parse_vlm_output(output, stop_str)
                            paths.append(path)
                            masks.append(mask)
                    failed = False

                except Exception as e:
                    logging.error(f"Error during batched VLM inference: {e}, retrying.... Outputs: {outputs}")

    return paths, masks


def generate_paths_masks(args: Args) -> None:
    """Generate paths and masks for Bridge data using direct VLM inference."""
    # Initialize model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, model_name, None
    )

    device = next(model.parameters()).device
    logging.info(
        f"Model loaded successfully on {device}. Context length: {context_len}"
    )

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load Bridge dataset
    raw_dataset = tfds.load("bridge_v2", data_dir=args.data_dir, split="train")

    # Create HDF5 file to store paths and masks
    h5_path = output_path / "bridge_paths_masks.h5"
    with h5py.File(h5_path, "w") as f:
        # Process each episode
        for episode_idx, episode in enumerate(
            tqdm.tqdm(raw_dataset, desc="Processing episodes")
        ):
            # Collect all images and task descriptions for this episode
            episode_images = []
            episode_tasks = []
            episode_timesteps = []
            episode_cameras = []  # Track which camera each image came from

            if not episode["episode_metadata"]["has_language"].numpy():
                logging.warning(
                    f"Episode {episode_idx} has no language instruction"
                )
                continue
            
            # Create group for this episode
            episode_group = f.create_group(f"episode_{episode_idx}")

            # Process each step
            for i, step in enumerate(episode["steps"].as_numpy_iterator()):
                if i % args.vlm_call_frequency != 0:
                    continue

                # Get task description for this step
                task_description = step["language_instruction"].decode()

                # Get images from all available cameras (not all zeros)
                step_images = []
                step_tasks = []
                step_timesteps = []
                step_cameras = []
                
                for cam in ["image_0", "image_1", "image_2", "image_3"]:
                    if cam in step["observation"]:
                        img = step["observation"][cam]
                        if img is not None and not np.all(img == 0):
                            step_images.append(img)
                            step_tasks.append(task_description)
                            step_timesteps.append(i)
                            step_cameras.append(cam)

                if not step_images:
                    logging.warning(
                        f"No valid images found in step {step['step_id']} of episode {episode_idx}"
                    )
                    continue

                # Add all valid images from this step to episode collections
                episode_images.extend(step_images)
                episode_tasks.extend(step_tasks)
                episode_timesteps.extend(step_timesteps)
                episode_cameras.extend(step_cameras)

            # Get task description
            episode_group.attrs["task_description"] = episode_tasks[0] if episode_tasks else ""

            if not episode_images:
                logging.warning(f"No valid images found in episode {episode_idx}")
                continue

            # Get paths and masks using direct VLM inference
            #try:
            paths, masks = get_path_mask_from_vlm_direct(
                episode_images,
                episode_tasks,
                model,
                tokenizer,
                image_processor,
                args,
                device,
            )

            # Save paths and masks for this episode
            if args.draw_path and paths:
                # Group paths by camera
                camera_paths = {}
                camera_path_timesteps = {}
                
                for i, (path, camera, timestep) in enumerate(zip(paths, episode_cameras, episode_timesteps)):
                    if camera not in camera_paths:
                        camera_paths[camera] = []
                        camera_path_timesteps[camera] = []
                    camera_paths[camera].append(path)
                    camera_path_timesteps[camera].append(timestep)
                
                # Save separate datasets for each camera
                for camera in camera_paths:
                    cam_paths = camera_paths[camera]
                    valid_paths = [p for p in cam_paths if p is not None and len(p) > 0]
                    
                    if valid_paths:
                        max_path_len = max(len(p) for p in valid_paths)
                        padded_paths = np.zeros((len(cam_paths), max_path_len, 2))
                        path_lengths = []
                        
                        for i, p in enumerate(cam_paths):
                            if p is not None and len(p) > 0:
                                padded_paths[i, : len(p)] = p
                                path_lengths.append(len(p))
                            else:
                                path_lengths.append(0)
                        
                        episode_group.create_dataset(f"{camera}_paths", data=padded_paths)
                        episode_group.create_dataset(f"{camera}_path_lengths", data=np.array(path_lengths))
                        episode_group.create_dataset(f"{camera}_path_timesteps", data=np.array(camera_path_timesteps[camera], dtype=np.uint16))

            if args.draw_mask and masks:
                # Group masks by camera
                camera_masks = {}
                camera_mask_timesteps = {}
                
                for i, (mask, camera, timestep) in enumerate(zip(masks, episode_cameras, episode_timesteps)):
                    if camera not in camera_masks:
                        camera_masks[camera] = []
                        camera_mask_timesteps[camera] = []
                    camera_masks[camera].append(mask)
                    camera_mask_timesteps[camera].append(timestep)
                
                # Save separate datasets for each camera
                for camera in camera_masks:
                    cam_masks = camera_masks[camera]
                    valid_masks = [m for m in cam_masks if m is not None and len(m) > 0]
                    
                    if valid_masks:
                        max_mask_len = max(len(m) for m in valid_masks)
                        padded_masks = np.zeros((len(cam_masks), max_mask_len, 2))
                        mask_lengths = []
                        
                        for i, m in enumerate(cam_masks):
                            if m is not None and len(m) > 0:
                                padded_masks[i, : len(m)] = m
                                mask_lengths.append(len(m))
                            else:
                                mask_lengths.append(0)
                        
                        episode_group.create_dataset(f"{camera}_masks", data=padded_masks)
                        episode_group.create_dataset(f"{camera}_mask_lengths", data=np.array(mask_lengths))
                        episode_group.create_dataset(f"{camera}_mask_timesteps", data=np.array(camera_mask_timesteps[camera], dtype=np.uint16))

            # Save some metadata
            episode_group.attrs["num_steps"] = len(episode_images)
            episode_group.attrs["has_paths"] = len(paths) > 0
            episode_group.attrs["has_masks"] = len(masks) > 0
            episode_group.attrs["task_description"] = episode_tasks[0] if episode_tasks else ""
            
            # Track which cameras have data
            unique_cameras = list(set(episode_cameras))
            episode_group.attrs["cameras_with_data"] = [cam.encode('utf-8') for cam in unique_cameras]
            episode_group.attrs["total_images"] = len(episode_images)

            #except Exception as e:
            #    logging.error(f"Error processing episode {episode_idx}: {e}")
            #    continue

    logging.info(f"Generated paths and masks saved to {h5_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(generate_paths_masks)
