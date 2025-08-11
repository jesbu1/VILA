"""
Script to generate paths and masks for the UW WidowX LeRobot dataset using direct VLM inference.

This mirrors the workflow in `label_fractal_data.py`, adapted to the Hugging Face
LeRobot dataset format with video frames per episode.

HDF5 layout per episode (same as other label_* scripts):

episode_group
    .attrs:
        task_description: str
        num_steps: uint16
        has_paths: bool
        has_masks: bool
        cameras_with_data: list[str]
        total_images: uint16

    Per-camera datasets (only created if camera has valid images):
    {camera}_paths: (num_images_for_camera, max_path_length, 2)
    {camera}_path_lengths: (num_images_for_camera)
    {camera}_path_timesteps: (num_images_for_camera)  # uint16
    {camera}_masks: (num_images_for_camera, max_mask_length, 2)
    {camera}_mask_lengths: (num_images_for_camera)
    {camera}_mask_timesteps: (num_images_for_camera)  # uint16

Usage example:

    # Pre-install deps:
    #   pip install datasets av h5py tyro tqdm torch
    #   pip install -e ~/vila_utils

    CUDA_VISIBLE_DEVICES=0 python label_uw_widowx_data.py \
        --args.output-dir ./uw_widowx_labels_3b \
        --args.model-path ~/.cache/huggingface/hub/models--memmelma--vila_3b_path_mask_fast/snapshots/12df7a04221a50e88733cd2f1132eb01257aba0d/checkpoint-11700/ \
        --args.vlm_call_frequency 50 \
        --args.batch-size 8

Notes:
- The dataset does not include natural language instructions; we synthesize a generic
  per-frame instruction using `task_index` to keep the VLM prompt consistent.
- MAKE SURE not installed/using torchcodec, as decoding lerobot dataset videos will fail since it depends on torchcodec which requires torch >=2.3, but this VILA hamster stuff uses torch==2.3.0.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
import tqdm
import tyro

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from data_labeling_utils import get_path_mask_from_vlm_direct
import dataclasses
from typing import Optional

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


DATASET_NAME = "jesbu1/uw_widowx_8_8_lerobot"
CAMERA_KEY = "images0"  # Matches feature name observation.images.images0


@dataclasses.dataclass
class Args:
    output_dir: str  # Directory to save the generated paths and masks
    model_path: str  # Path to the VLM model
    draw_path: bool = True  # Whether to generate paths
    draw_mask: bool = True  # Whether to generate masks
    flip_image_horizontally: bool = False  # Whether to flip images horizontally
    batch_size: int = 1  # Batch size for inference (1 for single, >1 for batched)
    temperature: float = 0.1
    top_p: Optional[float] = 0.95
    max_new_tokens: int = 512
    num_beams: int = 1
    vlm_call_frequency: int = 50  # Save every N timesteps
    load_8bit: bool = False
    max_retries: int = 3  # Maximum retries for accessing frames
    skip_problematic_frames: bool = (
        True  # Whether to skip frames that can't be accessed
    )


def _format_task_description(task_index: int) -> str:
    return (
        f"Task {task_index}. Draw the end-effector trajectory as a path and the manipulated "
        f"object as a mask in this image."
    )


def _get_instruction_for_frame(frame: Dict) -> str:
    try:
        task_index = int(frame.get("task_index", 0))
    except Exception:
        task_index = 0
    return _format_task_description(task_index)


def generate_paths_masks(args: Args) -> None:
    """Generate paths and masks for UW WidowX dataset using direct VLM inference."""
    # Initialize model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, model_name, None, load_8bit=args.load_8bit
    )

    device = next(model.parameters()).device
    model = torch.compile(model)  # type: ignore[arg-type]
    logging.info(
        f"Model loaded successfully on {device}. Context length: {context_len}"
    )

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset via LeRobot
    logging.info(f"Loading LeRobot dataset: {DATASET_NAME}")
    lerobot_dataset = LeRobotDataset(DATASET_NAME)

    # Test dataset accessibility
    def test_dataset_access():
        """Test if the dataset is accessible and identify potential issues."""
        try:
            logging.info("Testing dataset accessibility...")
            logging.info(f"Dataset length: {len(lerobot_dataset)}")
            logging.info(f"Number of episodes: {lerobot_dataset.num_episodes}")

            # Try to access the first few frames to test
            test_frames = min(5, len(lerobot_dataset))
            for i in range(test_frames):
                try:
                    frame = lerobot_dataset[i]
                    logging.info(f"Frame {i} accessible")
                except Exception as e:
                    logging.warning(f"Frame {i} not accessible: {e}")
                    break

        except Exception as e:
            logging.error(f"Dataset accessibility test failed: {e}")
            raise

    def check_dataset_integrity():
        """Check for potential dataset integrity issues."""
        try:
            logging.info("Checking dataset integrity...")

            # Check if episode data index is accessible
            if hasattr(lerobot_dataset, "episode_data_index"):
                logging.info("Episode data index accessible")
                logging.info(
                    f"Episode data index keys: {list(lerobot_dataset.episode_data_index.keys())}"
                )

                # Check a few episode ranges
                for i in range(min(3, lerobot_dataset.num_episodes)):
                    try:
                        from_idx = lerobot_dataset.episode_data_index["from"][i].item()
                        to_idx = lerobot_dataset.episode_data_index["to"][i].item()
                        logging.info(f"Episode {i}: frames {from_idx} to {to_idx}")
                    except Exception as e:
                        logging.warning(f"Episode {i} data index not accessible: {e}")
            else:
                logging.warning("Episode data index not accessible")

            # Check if underlying HuggingFace dataset is accessible
            if (
                hasattr(lerobot_dataset, "hf_dataset")
                and lerobot_dataset.hf_dataset is not None
            ):
                logging.info("Underlying HuggingFace dataset accessible")
                try:
                    logging.info(
                        f"HF dataset length: {len(lerobot_dataset.hf_dataset)}"
                    )
                    logging.info(
                        f"HF dataset features: {lerobot_dataset.hf_dataset.features}"
                    )
                except Exception as e:
                    logging.warning(f"HF dataset info not accessible: {e}")
            else:
                logging.warning("Underlying HuggingFace dataset not accessible")

        except Exception as e:
            logging.error(f"Dataset integrity check failed: {e}")

    # Run the tests
    test_dataset_access()
    check_dataset_integrity()

    # Helper function to safely get frame data
    def safe_get_frame(frame_idx: int, max_retries: int = None) -> Dict:
        """Safely get frame data with retry logic and error handling."""
        if max_retries is None:
            max_retries = args.max_retries

        for attempt in range(max_retries):
            try:
                # Try to access the frame directly
                frame = lerobot_dataset[frame_idx]
                return frame
            except RecursionError as e:
                logging.warning(
                    f"Recursion error on frame {frame_idx}, attempt {attempt + 1}: {e}"
                )
                if attempt == max_retries - 1:
                    # Try to access the underlying HuggingFace dataset directly as a last resort
                    try:
                        logging.info(
                            f"Attempting to access HuggingFace dataset directly for frame {frame_idx}"
                        )
                        # Access the underlying dataset directly to bypass the recursion issue
                        if (
                            hasattr(lerobot_dataset, "hf_dataset")
                            and lerobot_dataset.hf_dataset is not None
                        ):
                            # Try to get the raw data from the HuggingFace dataset
                            raw_data = lerobot_dataset.hf_dataset[frame_idx]
                            # Convert to the expected format if possible
                            if isinstance(raw_data, dict):
                                # Try to reconstruct the frame in the expected format
                                frame = {}
                                for key, value in raw_data.items():
                                    if key.startswith("observation.images."):
                                        frame[key] = value
                                    elif key == "task_index":
                                        frame[key] = value
                                    # Add other necessary keys as needed
                                if "observation.images." + CAMERA_KEY in frame:
                                    return frame
                        raise RuntimeError(
                            f"Failed to access frame {frame_idx} after {max_retries} attempts due to recursion error"
                        )
                    except Exception as fallback_error:
                        logging.error(
                            f"Fallback access also failed for frame {frame_idx}: {fallback_error}"
                        )
                        if args.skip_problematic_frames:
                            raise RuntimeError(
                                f"Frame {frame_idx} is problematic and will be skipped"
                            )
                        else:
                            raise RuntimeError(
                                f"Failed to access frame {frame_idx} after {max_retries} attempts due to recursion error"
                            )
                # Wait a bit before retrying
                import time

                time.sleep(0.1)
            except Exception as e:
                logging.warning(
                    f"Error accessing frame {frame_idx}, attempt {attempt + 1}: {e}"
                )
                if attempt == max_retries - 1:
                    if args.skip_problematic_frames:
                        raise RuntimeError(
                            f"Frame {frame_idx} is problematic and will be skipped"
                        )
                    else:
                        raise RuntimeError(
                            f"Failed to access frame {frame_idx} after {max_retries} attempts: {e}"
                        )
                # Wait a bit before retrying
                import time

                time.sleep(0.1)

        raise RuntimeError(f"Unexpected error: should not reach here")

    # Prepare HDF5
    h5_path = output_path / "uw_widowx_8_8_lerobot_paths_masks.h5"
    with h5py.File(h5_path, "a") as f:
        # Resume logic: determine last saved episode
        already_saved_episodes = sorted([int(k.split("_")[-1]) for k in f.keys()])
        if len(already_saved_episodes) == 0:
            last_episode = -1
        else:
            last_episode = already_saved_episodes[-1]

        for episode_idx in tqdm.tqdm(
            range(lerobot_dataset.num_episodes), desc="Processing episodes"
        ):
            try:
                # Skip completed episodes, drop the last incomplete
                if episode_idx < last_episode:
                    continue
                elif episode_idx == last_episode:
                    logging.info(f"Deleting episode {episode_idx} as it is incomplete")
                    try:
                        del f[f"episode_{episode_idx}"]
                    except KeyError as e:
                        logging.info(
                            f"Episode {episode_idx} not found, skipping its deletion: {e}"
                        )
                        continue

                # Determine frame range for this episode
                from_idx = lerobot_dataset.episode_data_index["from"][
                    episode_idx
                ].item()
                to_idx = lerobot_dataset.episode_data_index["to"][episode_idx].item()

                # Collect sampled images/tasks for this episode
                episode_images: List[np.ndarray] = []
                episode_tasks: List[str] = []
                episode_timesteps: List[int] = []
                episode_cameras: List[str] = []

                for i, frame_idx in enumerate(range(from_idx, to_idx)):
                    if i % max(1, args.vlm_call_frequency) != 0:
                        continue

                    try:
                        frame = safe_get_frame(frame_idx)
                        task_description = _get_instruction_for_frame(frame)

                        # Extract camera image from LeRobot sample: CxHxW float [0,1] -> HxWxC uint8
                        img_tensor = frame[f"observation.images.{CAMERA_KEY}"]
                        img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(
                            np.uint8
                        )

                        episode_images.append(img)
                        episode_tasks.append(task_description)
                        episode_timesteps.append(i)
                        episode_cameras.append(CAMERA_KEY)
                    except Exception as e:
                        logging.warning(
                            f"Failed to process frame {frame_idx} in episode {episode_idx}: {e}"
                        )
                        # Continue with the next frame instead of crashing
                        continue

                if not episode_images:
                    logging.warning(f"No selected frames for episode {episode_idx}")
                    continue

                # Create group for this episode
                episode_group = f.create_group(f"episode_{episode_idx}")

                # Save task description attribute (use first)
                episode_group.attrs["task_description"] = (
                    episode_tasks[0] if episode_tasks else ""
                )

                # VLM inference to get paths and masks
                try:
                    paths, masks = get_path_mask_from_vlm_direct(
                        episode_images,
                        episode_tasks,
                        model,
                        tokenizer,
                        image_processor,
                        args,
                        device,
                    )

                    # Save paths grouped by camera
                    if args.draw_path and paths:
                        camera_paths: Dict[str, List[np.ndarray]] = {}
                        camera_path_timesteps: Dict[str, List[int]] = {}
                        for path_arr, camera, timestep in zip(
                            paths, episode_cameras, episode_timesteps
                        ):
                            camera_paths.setdefault(camera, []).append(path_arr)
                            camera_path_timesteps.setdefault(camera, []).append(
                                timestep
                            )

                        for camera, cam_paths in camera_paths.items():
                            valid_paths = [
                                p for p in cam_paths if p is not None and len(p) > 0
                            ]
                            if valid_paths:
                                max_path_len = max(len(p) for p in valid_paths)
                                padded_paths = np.zeros(
                                    (len(cam_paths), max_path_len, 2)
                                )
                                path_lengths: List[int] = []
                                for i, p in enumerate(cam_paths):
                                    if p is not None and len(p) > 0:
                                        padded_paths[i, : len(p)] = p
                                        path_lengths.append(len(p))
                                    else:
                                        path_lengths.append(0)

                                episode_group.create_dataset(
                                    f"{camera}_paths", data=padded_paths
                                )
                                episode_group.create_dataset(
                                    f"{camera}_path_lengths",
                                    data=np.array(path_lengths),
                                )
                                episode_group.create_dataset(
                                    f"{camera}_path_timesteps",
                                    data=np.array(
                                        camera_path_timesteps[camera], dtype=np.uint16
                                    ),
                                )

                    # Save masks grouped by camera
                    if args.draw_mask and masks:
                        camera_masks: Dict[str, List[np.ndarray]] = {}
                        camera_mask_timesteps: Dict[str, List[int]] = {}
                        for mask_arr, camera, timestep in zip(
                            masks, episode_cameras, episode_timesteps
                        ):
                            camera_masks.setdefault(camera, []).append(mask_arr)
                            camera_mask_timesteps.setdefault(camera, []).append(
                                timestep
                            )

                        for camera, cam_masks in camera_masks.items():
                            valid_masks = [
                                m for m in cam_masks if m is not None and len(m) > 0
                            ]
                            if valid_masks:
                                max_mask_len = max(len(m) for m in valid_masks)
                                padded_masks = np.zeros(
                                    (len(cam_masks), max_mask_len, 2)
                                )
                                mask_lengths: List[int] = []
                                for i, m in enumerate(cam_masks):
                                    if m is not None and len(m) > 0:
                                        padded_masks[i, : len(m)] = m
                                        mask_lengths.append(len(m))
                                    else:
                                        mask_lengths.append(0)

                                episode_group.create_dataset(
                                    f"{camera}_masks", data=padded_masks
                                )
                                episode_group.create_dataset(
                                    f"{camera}_mask_lengths",
                                    data=np.array(mask_lengths),
                                )
                                episode_group.create_dataset(
                                    f"{camera}_mask_timesteps",
                                    data=np.array(
                                        camera_mask_timesteps[camera], dtype=np.uint16
                                    ),
                                )

                    # Metadata
                    episode_group.attrs["num_steps"] = len(episode_images)
                    episode_group.attrs["has_paths"] = bool(paths)
                    episode_group.attrs["has_masks"] = bool(masks)
                    episode_group.attrs["cameras_with_data"] = [
                        camera.encode("utf-8")
                        for camera in sorted(set(episode_cameras))
                    ]
                    episode_group.attrs["total_images"] = len(episode_images)

                except Exception as e:
                    logging.error(f"Error processing episode {episode_idx}: {e}")
                    del f[f"episode_{episode_idx}"]
                    continue
            except Exception as e:
                logging.error(f"Critical error processing episode {episode_idx}: {e}")
                # Try to clean up if the episode group was created
                try:
                    if f"episode_{episode_idx}" in f:
                        del f[f"episode_{episode_idx}"]
                except:
                    pass
                continue

    logging.info(f"Generated paths and masks saved to {h5_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(generate_paths_masks)
