import pandas as pd
import argparse
import os

if __name__ == "__main__":
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script using argparse')
    
    # Add the arguments
    parser.add_argument('--input_file', type=str, required=True, help='Input file')
    parser.add_argument('--predictions_file', type=str, required=True, help='Predictions file')
    parser.add_argument('--out_file', type=str, required=True, help='Out file')
    
    # Parse the arguments
    args = parser.parse_args()
    
    
    predictions = pd.read_csv(args.predictions_file, index_col=0)
    predictions = predictions.loc[predictions.predictions == 1]
    predictions = set(predictions.apply(lambda x: (int(x['left_ID']), int(x['right_ID'])), axis=1))
    
    edges = pd.read_csv(args.input_file)
    x = edges.shape
    edges = edges.loc[edges.apply(lambda x: (int(x['D1']), int(x['D2'])) in predictions, axis=1)]
    y = edges.shape
    print(x, y)
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    edges.to_csv(args.out_file, header=True, index=False)
    
    