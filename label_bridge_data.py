"""
Script to generate paths and masks for Bridge data using direct VLM inference.
This script should be run before convert_bridge_data_to_lerobot.py to prepare the path/mask data.
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

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


@dataclasses.dataclass
class Args:
    data_dir: str  # Directory containing the Bridge dataset
    output_dir: str  # Directory to save the generated paths and masks
    model_path: str  # Path to the VLM model
    conv_mode: str = "vicuna_v1"  # Conversation mode for VLM
    resize_size: int = 224  # Size to resize images for VLM
    draw_path: bool = True  # Whether to generate paths
    draw_mask: bool = True  # Whether to generate masks
    flip_image_horizontally: bool = False  # Whether to flip images horizontally
    batch_size: int = 1  # Batch size for inference (1 for single, >1 for batched)
    temperature: float = 0.2
    top_p: Optional[float] = 0.9
    max_new_tokens: int = 512
    num_beams: int = 1


def pad_sequence_to_max_length(seq, max_len, pad_token_id=0):
    """Pad sequence to maximum length."""
    if len(seq) >= max_len:
        return seq[:max_len]
    else:
        padding = torch.full(
            (max_len - len(seq),), pad_token_id, dtype=seq.dtype, device=seq.device
        )
        return torch.cat([seq, padding])


def normalize_image_tags(model, qs: str) -> str:
    """Normalize image tags in the query string."""
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = qs.replace(IMAGE_PLACEHOLDER, image_token_se)
        else:
            qs = qs.replace(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN)
    return qs


def get_path_mask_from_vlm_direct(
    images: List[np.ndarray],
    task_descriptions: List[str],
    model,
    tokenizer,
    image_processor,
    conv_mode: str,
    args: Args,
    device: str = "cuda",
) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    """
    Get path and mask predictions from VLM using direct inference.

    Args:
        images: List of image arrays
        task_descriptions: List of task descriptions
        model: Loaded VLM model
        tokenizer: Model tokenizer
        image_processor: Image processor
        conv_mode: Conversation mode
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
        # Convert to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            rgb_img = (
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.dtype == np.uint8 else img
            )
        else:
            rgb_img = img
        pil_image = Image.fromarray(rgb_img.astype(np.uint8))
        pil_images.append(pil_image)

    # Process in batches
    batch_size = min(args.batch_size, len(images))

    for i in range(0, len(images), batch_size):
        batch_images = pil_images[i : i + batch_size]
        batch_tasks = task_descriptions[i : i + batch_size]

        # Prepare conversation templates and prompts
        prompts = []
        batch_images_for_processing = []

        for j, (pil_img, task_desc) in enumerate(zip(batch_images, batch_tasks)):
            # Create conversation template
            conv = conv_templates[conv_mode].copy()
            user_role = conv.roles[0]
            assistant_role = conv.roles[1]

            # Create query for path and mask prediction
            if args.draw_path and args.draw_mask:
                query = f"Given the task: '{task_desc}', predict both the path and mask for this image. {IMAGE_PLACEHOLDER}"
            elif args.draw_path:
                query = f"Given the task: '{task_desc}', predict the path for this image. {IMAGE_PLACEHOLDER}"
            elif args.draw_mask:
                query = f"Given the task: '{task_desc}', predict the mask for this image. {IMAGE_PLACEHOLDER}"
            else:
                continue

            # Process the message similar to vila_3b_server.py message handling
            prompt = ""
            images_for_this_query = []

            # Add the image and text content
            prompt += query
            images_for_this_query.append(pil_img)

            # Normalize image tags
            normalized_prompt = normalize_image_tags(model, prompt)

            # Add messages to conversation
            conv.append_message(user_role, normalized_prompt)
            conv.append_message(assistant_role, None)

            # Get the full prompt
            prompt_text = conv.get_prompt()
            prompts.append(prompt_text)
            batch_images_for_processing.extend(images_for_this_query)

        if not prompts:
            # Add empty results for this batch
            for _ in range(len(batch_images)):
                paths.append(None)
                masks.append(None)
            continue

        try:
            # Process images
            images_tensor = process_images(
                batch_images_for_processing, image_processor, model.config
            ).to(device, dtype=torch.float16)

            # For batched inference, we need to handle multiple images properly
            if len(batch_images_for_processing) > 1:
                # Split images tensor for each prompt
                images_input = [
                    images_tensor[j : j + 1]
                    for j in range(len(batch_images_for_processing))
                ]
            else:
                images_input = [images_tensor]

            # Tokenize prompts and pad to same length
            batch_input_ids = [
                tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .squeeze(0)
                .to(device)
                for prompt in prompts
            ]

            if len(batch_input_ids) > 1:
                # Batched inference - pad sequences
                max_len = max([len(seq) for seq in batch_input_ids])
                padded_input_ids = [
                    pad_sequence_to_max_length(
                        seq, max_len, tokenizer.pad_token_id or 0
                    )
                    for seq in batch_input_ids
                ]
                batch_input_ids = torch.stack(padded_input_ids)

                # For batched inference, combine all images
                combined_images_tensor = torch.cat(images_input, dim=0)
                images_input_final = combined_images_tensor
            else:
                # Single inference
                batch_input_ids = batch_input_ids[0].unsqueeze(0)
                images_input_final = images_input[0]

            # Set up stopping criteria
            conv_temp = conv_templates[conv_mode].copy()
            stop_str = conv_temp.sep if hasattr(conv_temp, "sep") else None
            keywords = [stop_str] if stop_str else []
            stopping_criteria = (
                KeywordsStoppingCriteria(keywords, tokenizer, batch_input_ids)
                if keywords
                else []
            )

            # Generate responses
            with torch.inference_mode():
                output_ids = model.generate(
                    batch_input_ids,
                    images=images_input_final,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=stopping_criteria if stopping_criteria else None,
                )

            # Decode outputs
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            # Parse outputs to extract path and mask information
            for output in outputs:
                output = output.strip()
                if stop_str and output.endswith(stop_str):
                    output = output[: -len(stop_str)].strip()

                # Parse the output to extract path and mask coordinates
                # This is a simplified parser - you may need to adapt based on your model's output format
                path_coords = None
                mask_coords = None

                # Example parsing (adapt based on your model's output format)
                if "path:" in output.lower():
                    # Extract path coordinates from output
                    # This is a placeholder - implement based on your model's output format
                    path_coords = np.array([])  # Replace with actual parsing

                if "mask:" in output.lower():
                    # Extract mask coordinates from output
                    # This is a placeholder - implement based on your model's output format
                    mask_coords = np.array([])  # Replace with actual parsing

                paths.append(path_coords)
                masks.append(mask_coords)

        except Exception as e:
            logging.error(f"Error during VLM inference: {e}")
            # Add None results for this batch
            for _ in range(len(batch_images)):
                paths.append(None)
                masks.append(None)

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
            # Create group for this episode
            episode_group = f.create_group(f"episode_{episode_idx}")

            # Get task description
            task_description = episode["language_instruction"].decode()
            episode_group.attrs["task_description"] = task_description

            # Collect all images and task descriptions for this episode
            episode_images = []
            episode_tasks = []

            # Process each step
            for step in episode["steps"].as_numpy_iterator():
                # Get image from first available camera
                img = None
                for cam in ["image_0", "image_1", "image_2", "image_3"]:
                    if cam in step["observation"]:
                        img = step["observation"][cam]
                        break

                if img is None:
                    logging.warning(
                        f"No valid image found in step {step['step_id']} of episode {episode_idx}"
                    )
                    continue

                episode_images.append(img)
                episode_tasks.append(task_description)

            if not episode_images:
                logging.warning(f"No valid images found in episode {episode_idx}")
                continue

            # Get paths and masks using direct VLM inference
            try:
                paths, masks = get_path_mask_from_vlm_direct(
                    episode_images,
                    episode_tasks,
                    model,
                    tokenizer,
                    image_processor,
                    args.conv_mode,
                    args,
                    str(device),
                )

                # Save paths and masks for this episode
                if args.draw_path and paths:
                    valid_paths = [p for p in paths if p is not None and len(p) > 0]
                    if valid_paths:
                        max_path_len = max(len(p) for p in valid_paths)
                        padded_paths = np.zeros((len(paths), max_path_len, 2))
                        path_lengths = []
                        for i, p in enumerate(paths):
                            if p is not None and len(p) > 0:
                                padded_paths[i, : len(p)] = p
                                path_lengths.append(len(p))
                            else:
                                path_lengths.append(0)
                        episode_group.create_dataset("paths", data=padded_paths)
                        episode_group.create_dataset(
                            "path_lengths", data=np.array(path_lengths)
                        )

                if args.draw_mask and masks:
                    valid_masks = [m for m in masks if m is not None and len(m) > 0]
                    if valid_masks:
                        max_mask_len = max(len(m) for m in valid_masks)
                        padded_masks = np.zeros((len(masks), max_mask_len, 2))
                        mask_lengths = []
                        for i, m in enumerate(masks):
                            if m is not None and len(m) > 0:
                                padded_masks[i, : len(m)] = m
                                mask_lengths.append(len(m))
                            else:
                                mask_lengths.append(0)
                        episode_group.create_dataset("masks", data=padded_masks)
                        episode_group.create_dataset(
                            "mask_lengths", data=np.array(mask_lengths)
                        )

                # Save some metadata
                episode_group.attrs["num_steps"] = len(episode_images)
                episode_group.attrs["has_paths"] = args.draw_path
                episode_group.attrs["has_masks"] = args.draw_mask

            except Exception as e:
                logging.error(f"Error processing episode {episode_idx}: {e}")
                continue

    logging.info(f"Generated paths and masks saved to {h5_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(generate_paths_masks)
