import os
import h5py
import numpy as np
import argparse


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

    merged_data = {}
    dataset_shapes = {}

    # First pass to collect and concatenate data
    for idx, file_path in enumerate(h5_files):
        with h5py.File(file_path, "r") as f:
            for key in f.keys():
                data = f[key][:]
                if idx == 0:
                    merged_data[key] = [data]
                else:
                    merged_data[key].append(data)

    # Concatenate all the data
    for key in merged_data:
        merged_data[key] = np.concatenate(merged_data[key], axis=0)

    # Write to output
    with h5py.File(output_path, "w") as f_out:
        for key, data in merged_data.items():
            f_out.create_dataset(key, data=data)

    print(f"Merged {len(h5_files)} files into '{output_path}'.")


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple HDF5 files into one.")
    parser.add_argument(
        "input_directory",
        type=str,
        help="The directory containing the .h5 or .hdf5 files to be merged.",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="The path to the output merged .h5 file.",
    )
    args = parser.parse_args()
    merge_h5_files(args.input_directory, args.output_file)
