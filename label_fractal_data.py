"""
Script to generate paths and masks for Libero data using direct VLM inference.
This script should be run before convert_libero_data_to_lerobot.py to prepare the path/mask data.

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


    #--args.model-path ~/.cache/huggingface/hub/models--memmelma--vila_3b_path_mask_fast/snapshots/12df7a04221a50e88733cd2f1132eb01257aba0d/checkpoint-11700/ \
CUDA_VISIBLE_DEVICES=0 python label_fractal_data.py \
    --args.data-dir=/data/shared/openx_rlds_data/ \
    --args.output-dir=./fractal_labels_3b \
    --args.model-path ~/.cache/huggingface/hub/models--memmelma--vila_3b_path_mask_fast/snapshots/12df7a04221a50e88733cd2f1132eb01257aba0d/checkpoint-11700/ \
    --args.vlm_call_frequency=50 \
    --args.batch-size=8 
"""

import logging
from pathlib import Path
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import tqdm
import tyro
import h5py
from data_labeling_utils import Args, get_path_mask_from_vlm_direct


from llava.mm_utils import (
    get_model_name_from_path,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


RAW_DATASET_NAMES = ["fractal20220817_data"]


def generate_paths_masks(args: Args) -> None:
    """Generate paths and masks for Libero data using direct VLM inference."""
    # Initialize model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, model_name, None, load_8bit=args.load_8bit
    )

    device = next(model.parameters()).device
    logging.info(
        f"Model loaded successfully on {device}. Context length: {context_len}"
    )

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create HDF5 file to store paths and masks
    for raw_dataset_name in RAW_DATASET_NAMES:
        # Load dataset builder
        builder = tfds.builder(raw_dataset_name, data_dir=args.data_dir)
        # Define custom decoding behavior for the field that expects 256 x 320 for some reason when
        # it should be 256 x 256
        custom_decoder = {
            "steps": {
                "observation": {
                    "image": tfds.decode.SkipDecoding(),  # We'll decode manually later
                }
            }
        }

        # Load the dataset, skipping strict shape checks
        raw_dataset = builder.as_dataset(split="train", decoders=custom_decoder)

        h5_path = output_path / f"{raw_dataset_name}_paths_masks.h5"
        with h5py.File(h5_path, "a") as f:
            # Process each episode. Resume by going to the second from last episode.
            already_saved_episodes = sorted([int(k.split("_")[-1]) for k in f.keys()])
            if len(already_saved_episodes) == 0:
                last_episode = -1
            else:
                last_episode = already_saved_episodes[-1]
            for episode_idx, episode in enumerate(
                tqdm.tqdm(raw_dataset, desc="Processing episodes")
            ):
                # Skip episodes that have already been saved
                if episode_idx < last_episode:
                    continue
                # Delete the last episode if it exists, as it might be incomplete
                elif episode_idx == last_episode:
                    logging.info(f"Deleting episode {episode_idx} as it is incomplete")
                    try:
                        del f[f"episode_{episode_idx}"]
                    except KeyError as e:
                        logging.info(
                            f"Episode {episode_idx} not found, skipping its deletion: {e}"
                        )
                        breakpoint()
                        continue

                # Collect all images and task descriptions for this episode
                episode_images = []
                episode_tasks = []
                episode_timesteps = []
                episode_cameras = []  # Track which camera each image came from

                # Create group for this episode
                episode_group = f.create_group(f"episode_{episode_idx}")

                # Process each step
                for i, step in enumerate(episode["steps"]):
                    if i % args.vlm_call_frequency != 0:
                        continue

                    # Get task description for this step
                    task_description = (
                        step["observation"]["natural_language_instruction"]
                        .numpy()
                        .decode()
                    )

                    # Get images from all available cameras (not all zeros)
                    step_images = []
                    step_tasks = []
                    step_timesteps = []
                    step_cameras = []
                    for cam in ["image"]:
                        if cam in step["observation"]:
                            img = tf.io.decode_image(step["observation"][cam]).numpy()
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
                episode_group.attrs["task_description"] = (
                    episode_tasks[0] if episode_tasks else ""
                )

                if not episode_images:
                    logging.warning(f"No valid images found in episode {episode_idx}")
                    continue

                # Get paths and masks using direct VLM inference
                try:

                    # for history
                    # all_paths = []
                    # all_masks = []
                    # for episode_step in range(len(episode_images)):
                    #    if len(all_paths) > 0:
                    #        path_history = all_paths[-1:]
                    #    else:
                    #        path_history = None
                    #    paths, masks = get_path_mask_from_vlm_direct(
                    #        episode_images[episode_step:episode_step+1],
                    #        episode_tasks[episode_step:episode_step+1],
                    #        model,
                    #        tokenizer,
                    #        image_processor,
                    #        args,
                    #        device,
                    #        path_history=path_history,
                    #    )
                    #    all_paths.extend(paths)
                    #    all_masks.extend(masks)
                    # paths = all_paths
                    # masks = all_masks

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

                        for i, (path, camera, timestep) in enumerate(
                            zip(paths, episode_cameras, episode_timesteps)
                        ):
                            if camera not in camera_paths:
                                camera_paths[camera] = []
                                camera_path_timesteps[camera] = []
                            camera_paths[camera].append(path)
                            camera_path_timesteps[camera].append(timestep)

                        # Save separate datasets for each camera
                        for camera in camera_paths:
                            cam_paths = camera_paths[camera]
                            valid_paths = [
                                p for p in cam_paths if p is not None and len(p) > 0
                            ]

                            if valid_paths:
                                max_path_len = max(len(p) for p in valid_paths)
                                padded_paths = np.zeros(
                                    (len(cam_paths), max_path_len, 2)
                                )
                                path_lengths = []

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

                    if args.draw_mask and masks:
                        # Group masks by camera
                        camera_masks = {}
                        camera_mask_timesteps = {}

                        for i, (mask, camera, timestep) in enumerate(
                            zip(masks, episode_cameras, episode_timesteps)
                        ):
                            if camera not in camera_masks:
                                camera_masks[camera] = []
                                camera_mask_timesteps[camera] = []
                            camera_masks[camera].append(mask)
                            camera_mask_timesteps[camera].append(timestep)

                        # Save separate datasets for each camera
                        for camera in camera_masks:
                            cam_masks = camera_masks[camera]
                            valid_masks = [
                                m for m in cam_masks if m is not None and len(m) > 0
                            ]

                            if valid_masks:
                                max_mask_len = max(len(m) for m in valid_masks)
                                padded_masks = np.zeros(
                                    (len(cam_masks), max_mask_len, 2)
                                )
                                mask_lengths = []

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

                    # Save some metadata
                    episode_group.attrs["num_steps"] = len(episode_images)
                    episode_group.attrs["has_paths"] = len(paths) > 0
                    episode_group.attrs["has_masks"] = len(masks) > 0
                    episode_group.attrs["task_description"] = (
                        episode_tasks[0] if episode_tasks else ""
                    )

                    # Track which cameras have data
                    unique_cameras = list(set(episode_cameras))
                    episode_group.attrs["cameras_with_data"] = [
                        cam.encode("utf-8") for cam in unique_cameras
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
