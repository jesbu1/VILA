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
        --args.vlm_call_frequency 5 \
        --args.batch-size 8

Notes:
- The dataset does not include natural language instructions; we synthesize a generic
  per-frame instruction using `task_index` to keep the VLM prompt consistent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
import tqdm
import tyro

import av  # PyAV for decoding video frames
from datasets import load_dataset

from data_labeling_utils import Args, get_path_mask_from_vlm_direct

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


DATASET_NAME = "jesbu1/uw_widowx_8_8_lerobot"
CAMERA_KEY = "images0"  # Matches feature name observation.images.images0


def _format_task_description(task_index: int) -> str:
    return (
        f"Task {task_index}. Draw the end-effector trajectory as a path and the manipulated "
        f"object as a mask in this image."
    )


def _collect_episode_indices(ds) -> Dict[int, List[int]]:
    """Group dataset row indices by episode_index.

    The dataset is small (~2.5k rows), so a single pass is fine.
    """
    episode_to_indices: Dict[int, List[int]] = {}
    for i in range(len(ds)):
        ep_idx = int(ds[i]["episode_index"])  # type: ignore[index]
        episode_to_indices.setdefault(ep_idx, []).append(i)
    # Ensure indices in each episode are ordered by frame_index
    for ep_idx, idxs in episode_to_indices.items():
        idxs.sort(key=lambda j: int(ds[j]["frame_index"]))
    return episode_to_indices


def _get_video_path(sample) -> str:
    """Extract the local video path from a dataset sample's Video feature.

    The `observation.images.images0` field is a Video feature that resolves to a mapping
    containing a local `path` key once materialized by `datasets`.
    """
    vid_field = sample["observation"]["images"][CAMERA_KEY]
    if isinstance(vid_field, dict) and "path" in vid_field:
        return vid_field["path"]
    if isinstance(vid_field, str):
        return vid_field
    raise ValueError("Unexpected video field structure; cannot locate video path")


def _decode_selected_frames(
    video_path: str, desired_frame_indices: List[int]
) -> List[np.ndarray]:
    """Decode selected frames from a video using PyAV.

    Returns a list of HxWxC uint8 RGB arrays corresponding to the desired frame indices,
    in ascending frame index order.
    """
    if not desired_frame_indices:
        return []
    desired_set = set(desired_frame_indices)
    max_desired = max(desired_frame_indices)
    images_by_index: Dict[int, np.ndarray] = {}

    with av.open(video_path) as container:
        video_stream = container.streams.video[0]
        frame_idx = 0
        for packet in container.demux(video_stream):
            for frame in packet.decode():
                if frame_idx in desired_set:
                    img = frame.to_ndarray(format="rgb24")  # HxWx3, uint8
                    images_by_index[frame_idx] = img
                    if len(images_by_index) == len(desired_set):
                        break
                frame_idx += 1
            if len(images_by_index) == len(desired_set) or frame_idx > max_desired:
                break

    # Return in the order of desired_frame_indices
    return [images_by_index[i] for i in desired_frame_indices if i in images_by_index]


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

    # Load dataset from Hugging Face Hub
    logging.info(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME, split="train")

    # Prepare HDF5
    h5_path = output_path / "uw_widowx_8_8_lerobot_paths_masks.h5"
    with h5py.File(h5_path, "a") as f:
        # Resume logic: determine last saved episode
        already_saved_episodes = sorted([int(k.split("_")[-1]) for k in f.keys()])
        if len(already_saved_episodes) == 0:
            last_episode = -1
        else:
            last_episode = already_saved_episodes[-1]

        # Build mapping from episode_index to row indices
        episode_to_indices = _collect_episode_indices(ds)

        for episode_idx in tqdm.tqdm(
            sorted(episode_to_indices.keys()), desc="Processing episodes"
        ):
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

            row_indices = episode_to_indices[episode_idx]
            if len(row_indices) == 0:
                logging.warning(f"No rows found for episode {episode_idx}")
                continue

            # Sample frames by frequency and collect metadata
            desired_rows: List[int] = []
            desired_frame_indices: List[int] = []
            task_descriptions: List[str] = []
            for row_idx in row_indices:
                row = ds[row_idx]
                frame_index = int(row["frame_index"])  # type: ignore[index]
                if frame_index % max(1, args.vlm_call_frequency) != 0:
                    continue
                desired_rows.append(row_idx)
                desired_frame_indices.append(frame_index)
                task_index = int(row["task_index"])  # type: ignore[index]
                task_descriptions.append(_format_task_description(task_index))

            if not desired_rows:
                logging.warning(f"No selected frames for episode {episode_idx}")
                continue

            # Video path (assume same across episode)
            sample0 = ds[desired_rows[0]]
            try:
                video_path = _get_video_path(sample0)
            except Exception as e:
                logging.error(
                    f"Failed to resolve video path for episode {episode_idx}: {e}"
                )
                continue

            # Decode frames
            episode_images = _decode_selected_frames(video_path, desired_frame_indices)
            if not episode_images:
                logging.warning(f"No frames decoded for episode {episode_idx}")
                continue

            # Camera/timestep tracking
            episode_cameras = [CAMERA_KEY for _ in episode_images]
            episode_timesteps = desired_frame_indices

            # Create group for this episode
            episode_group = f.create_group(f"episode_{episode_idx}")

            # Save task description attribute (use first)
            episode_group.attrs["task_description"] = (
                task_descriptions[0] if task_descriptions else ""
            )

            # VLM inference to get paths and masks
            try:
                paths, masks = get_path_mask_from_vlm_direct(
                    episode_images,
                    task_descriptions,
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
                        camera_path_timesteps.setdefault(camera, []).append(timestep)

                    for camera, cam_paths in camera_paths.items():
                        valid_paths = [
                            p for p in cam_paths if p is not None and len(p) > 0
                        ]
                        if valid_paths:
                            max_path_len = max(len(p) for p in valid_paths)
                            padded_paths = np.zeros((len(cam_paths), max_path_len, 2))
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
                                f"{camera}_path_lengths", data=np.array(path_lengths)
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
                        camera_mask_timesteps.setdefault(camera, []).append(timestep)

                    for camera, cam_masks in camera_masks.items():
                        valid_masks = [
                            m for m in cam_masks if m is not None and len(m) > 0
                        ]
                        if valid_masks:
                            max_mask_len = max(len(m) for m in valid_masks)
                            padded_masks = np.zeros((len(cam_masks), max_mask_len, 2))
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
                                f"{camera}_mask_lengths", data=np.array(mask_lengths)
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
                    camera.encode("utf-8") for camera in sorted(set(episode_cameras))
                ]
                episode_group.attrs["total_images"] = len(episode_images)

            except Exception as e:
                logging.error(f"Error processing episode {episode_idx}: {e}")
                del f[f"episode_{episode_idx}"]
                continue

    logging.info(f"Generated paths and masks saved to {h5_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(generate_paths_masks)
