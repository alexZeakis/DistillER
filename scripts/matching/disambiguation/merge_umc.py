import pandas as pd
import argparse
import os

# def keep(row, cleaned):
#     if row['predictions'] == 0:
#         return True
#     else:
#         if (int(row['left_ID']), int(row['right_ID'])) in cleaned:
#             return True
#     return False

def keep(row, cleaned):
    if row['predictions'] == 0:
        return 0
    else:
        if (int(row['left_ID']), int(row['right_ID'])) in cleaned:
            return 1
    return 0


if __name__ == "__main__":
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script using argparse')
    
    # Add the arguments
    parser.add_argument('--input_file', type=str, required=True, help='Input file')
    parser.add_argument('--cleaned_file', type=str, required=True, help='Cleaned of UMC file')
    parser.add_argument('--out_file', type=str, required=True, help='Out file')
    
    # Parse the arguments
    args = parser.parse_args()
    
    cleaned = pd.read_csv(args.cleaned_file)
    cleaned = set(cleaned.apply(lambda x: (x['D1'], x['D2']), axis=1))
    
    edges = pd.read_csv(args.input_file, index_col=0)
    # edges = edges.loc[edges.apply(lambda x: keep(x, cleaned), axis=1)]
    edges.predictions = edges.apply(lambda x: keep(x, cleaned), axis=1)
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    edges.to_csv(args.out_file, header=True, index=True)
    
    