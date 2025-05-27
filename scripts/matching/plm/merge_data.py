import argparse
import os
import pandas as pd

def merge_csvs(in_dir, out_dir):
    split_names = ["train", "valid", "test"]
    os.makedirs(out_dir, exist_ok=True)

    for split in split_names:
        
        files = [os.path.join(in_dir, data_dir, f"{split}.csv")
                     for data_dir in os.listdir(in_dir) if data_dir!='total']
        
        df = pd.DataFrame()
        for f in files:
            try:
                df = pd.concat([df, pd.read_csv(f)], ignore_index=True)
            except Exception as e:
                print(f"Could not read {f}: {e}")

        output_path = os.path.join(out_dir, f"{split}.csv")
        df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate train/valid/test CSVs from subdirs.")
    parser.add_argument("--in_dir", help="Path to the parent directory containing subdirectories with CSVs")
    parser.add_argument("--out_dir", help="Name of the output directory")
    args = parser.parse_args()

    merge_csvs(args.in_dir, args.out_dir)
