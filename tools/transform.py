import os
import lmdb
import argparse
from tqdm import tqdm

def convert_simple_to_lmdb(data_root_dir, label_file_paths, lmdb_output_dir):
    """
    Convert a SimpleDataSet to an LMDBDataSet in PaddleOCR format.

    Args:
        data_root_dir (str): The base directory containing the images.
        label_file_paths (list of str): List of paths to the label files.
        lmdb_output_dir (str): The directory where the LMDB database will be created.

    Note:
        - The label files should contain lines with 'relative_path\tlabel', where relative_path is relative to data_root_dir.
        - Creates an LMDB database at lmdb_output_dir with keys 'image-000000000', 'label-000000000', etc.
        - Requires 'lmdb' library; install with 'pip install lmdb'.
        - After conversion, use in PaddleOCR config: Train.dataset.name: LMDBDataSet, Train.dataset.data_dir: lmdb_output_dir
    """
    # Count total number of lines for progress bar (may overestimate due to invalid lines)
    total_lines = 0
    for label_file in label_file_paths:
        with open(label_file, 'r', encoding='utf-8') as f:
            total_lines += sum(1 for _ in f)
    
    # Create LMDB environment with a large map size (1TB) to accommodate most datasets
    env = lmdb.open(lmdb_output_dir, map_size=1099511627776)  # 1TB
    
    # Write data to LMDB incrementally
    index = 1
    with env.begin(write=True) as txn:
        pbar = tqdm(total=total_lines, desc="Converting to LMDB")
        for label_file in label_file_paths:
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) != 2:
                        pbar.update(1)
                        continue
                    rel_path, label = parts
                    full_path = os.path.join(data_root_dir, rel_path)
                    if not os.path.exists(full_path):
                        print(f"Warning: Image not found: {full_path}")
                        pbar.update(1)
                        continue
                    with open(full_path, 'rb') as img_f:
                        img_data = img_f.read()
                    label_bytes = label.encode('utf-8')
                    img_key = f'image-{index:09d}'.encode()
                    label_key = f'label-{index:09d}'.encode()
                    txn.put(img_key, img_data)
                    txn.put(label_key, label_bytes)
                    index += 1
                    pbar.update(1)
        # Store the total number of samples
        num_samples_key = b'num-samples'
        txn.put(num_samples_key, str(index).encode())
        pbar.close()
    
    env.close()

def main():
    parser = argparse.ArgumentParser(description="Convert PaddleOCR SimpleDataSet to LMDBDataSet format.")
    parser.add_argument(
        "--data-root-dir",
        type=str,
        required=True,
        help="Base directory containing the images (e.g., './train_data/')."
    )
    parser.add_argument(
        "--label-files",
        type=str,
        nargs='+',
        required=True,
        help="Paths to label files (e.g., './train_data/ihr-train.txt'). Multiple files can be specified."
    )
    parser.add_argument(
        "--lmdb-output-dir",
        type=str,
        required=True,
        help="Directory where the LMDB database will be created (e.g., './train_data/lmdb_train/')."
    )
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.data_root_dir):
        raise ValueError(f"Data root directory does not exist: {args.data_root_dir}")
    for label_file in args.label_files:
        if not os.path.exists(label_file):
            raise ValueError(f"Label file does not exist: {label_file}")
    if not os.path.exists(args.lmdb_output_dir):
        os.makedirs(args.lmdb_output_dir)

    # Run conversion
    convert_simple_to_lmdb(args.data_root_dir, args.label_files, args.lmdb_output_dir)
    print(f"LMDB dataset created at: {args.lmdb_output_dir}")

if __name__ == "__main__":
    """
    python convert_simple_to_lmdb.py \
        --data-root-dir ./train_data/ \
        --label-files ./train_data/ihr-train.txt ./train_data/ihr-val.txt ./train_data/kinh_su_tu.txt ./train_data/nomna-train.txt ./train_data/hisdoc1b_500k.txt ./train_data/all.txt \
        --lmdb-output-dir ./train_data/rec/
    """
    main()