import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def merge_csvs(in_dir, out_dir, split):
    files = [os.path.join(in_dir, data_dir, f"{split}.csv")
                 for data_dir in os.listdir(in_dir) if data_dir!='total']
    
    df = pd.DataFrame()
    for f in files:
        try:
            df = pd.concat([df, pd.read_csv(f)], ignore_index=True)
        except Exception as e:
            print(f"Could not read {f}: {e}")

    print(df.shape)
    return df

    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate train/valid/test CSVs from subdirs.")
    parser.add_argument("--in_dir", help="Path to the parent directory containing subdirectories with CSVs")
    parser.add_argument("--out_dir", help="Name of the output directory")
    parser.add_argument("--percentage", type=float, default=0.8, help="Percentage of splitting validation data.")
    args = parser.parse_args()


    os.makedirs(args.out_dir, exist_ok=True)

    train_df = merge_csvs(args.in_dir, args.out_dir, "train")
    train_df.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)
    
    valid_df = merge_csvs(args.in_dir, args.out_dir, "valid")
    train_df, valid_df = train_test_split(
        valid_df,
        test_size=1-args.percentage,
        stratify=valid_df.Label,
        random_state=1924  # for reproducibility
    )
    
    train_df.to_csv(os.path.join(args.out_dir, "train.csv"), header=True, index=False)
    valid_df.to_csv(os.path.join(args.out_dir, "valid.csv"), header=True, index=False)