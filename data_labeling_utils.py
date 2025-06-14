import torch
import logging
import numpy as np
import cv2
CONV_MODE = "vicuna_v1"
from vila_utils.utils.prompts import get_prompt
from vila_utils.utils.decode import (
    add_path_2d_to_img_alt_fast,
    add_mask_2d_to_img,
    get_path_from_answer,
)
PROMPT_TYPE = "path_mask"
from vila_utils.utils.encode import scale_path
from PIL import Image
from typing import List, Tuple, Optional
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
import dataclasses

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
            failure_count = 0
            while failed and failure_count < 5:
                try:
                    # Create conversation template (following vila_3b_server.py pattern)
                    conv = conv_templates[CONV_MODE].copy()
                    user_role = conv.roles[0]
                    assistant_role = conv.roles[1]

                    # Create query for path and mask prediction
                    #query = get_prompt(task_desc, PROMPT_TYPE, prompt_eval=True)
                    query = get_prompt(task_desc, PROMPT_TYPE)
                    #query = f"{IMAGE_PLACEHOLDER}{query}"

                    if query is None:
                        paths.append(None)
                        masks.append(None)
                        continue

                    # Normalize image tags
                    #normalized_query = normalize_image_tags(model, query)

                    # Add messages to conversation
                    #conv.append_message(user_role, normalized_query)
                    conv.append_message(user_role, query)
                    conv.append_message(assistant_role, None)

                    # Get the full prompt
                    prompt_text = conv.get_prompt()
                    breakpoint()

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
                    failure_count += 1
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
            failure_count = 0
            while failed and failure_count < 5:

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
                        assistant_role = conv.roles[1]

                        # Create query for path and mask prediction
                        #query = get_prompt(task_desc, PROMPT_TYPE, prompt_eval=True)
                        #query = f"{IMAGE_PLACEHOLDER}{query}"
                        query = get_prompt(task_desc, PROMPT_TYPE)

                        if query is None:
                            batch_results_map.append(None)  # Mark as skipped
                            continue

                        # Normalize image tags and build conversation
                        #normalized_query = normalize_image_tags(model, query)
                        #conv.append_message(user_role, normalized_query)
                        conv.append_message(user_role, query)
                        conv.append_message(assistant_role, None)

                        prompt_text = conv.get_prompt()
                        print(prompt_text)
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
                    failure_count += 1
                    logging.error(f"Error during batched VLM inference: {e}, retrying.... Outputs: {outputs}")
                    if failure_count >= 5:
                        raise e

    return paths, masks

