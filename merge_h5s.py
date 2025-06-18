import os
import h5py
import numpy as np
import argparse

def merge_attrs(source, target):
    for key, value in source.attrs.items():
        target.attrs[key] = value  # Overwrites if key exists


def recursive_merge(source_group, target_group):
    for key in source_group:
        source_item = source_group[key]

        if isinstance(source_item, h5py.Group):
            # Create group if it doesn't exist
            if key not in target_group:
                target_subgroup = target_group.create_group(key)
            else:
                target_subgroup = target_group[key]
            # Merge attributes
            merge_attrs(source_item, target_subgroup)
            # Recurse
            recursive_merge(source_item, target_subgroup)

        elif isinstance(source_item, h5py.Dataset):
            if key in target_group:
                # Concatenate datasets
                existing_data = target_group[key][()]
                new_data = source_item[()]
                combined_data = np.concatenate((existing_data, new_data), axis=0)

                # Delete and recreate with new size
                del target_group[key]
                dset = target_group.create_dataset(key, data=combined_data)
                merge_attrs(source_item, dset)
            else:
                # Just copy the dataset
                dset = target_group.create_dataset(key, data=source_item[()])
                merge_attrs(source_item, dset)


def merge_h5_files(input_dir, output_path):
    h5_files = sorted(
        [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".h5") or f.endswith(".hdf5")
        ]
    )

    if not h5_files:
        raise ValueError("No .h5 files found in the directory.")

    with h5py.File(output_path, "w") as f_out:
        for idx, file_path in enumerate(h5_files):
            with h5py.File(file_path, "r") as f_in:
                recursive_merge(f_in, f_out)

    print(f"Merged {len(h5_files)} files into '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively merge multiple HDF5 files into one.")
    parser.add_argument(
        "--input-directory",
        type=str,
        required=True,
        help="Directory containing the .h5 or .hdf5 files to merge.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to output merged .h5 file.",
    )
    args = parser.parse_args()
    merge_h5_files(args.input_directory, args.output_file)
