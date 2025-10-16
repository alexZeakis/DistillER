import pandas as pd
from time import time
import os
import argparse

def evaluate(ground_results, results):
    true_positives = len(ground_results & results)
    if true_positives > 0:
        recall =  true_positives / len(ground_results)
        precision =  true_positives / len(results)
        f1 = 2 * (precision*recall) / (precision + recall)
    else:
        recall, precision, f1 = 0, 0, 0
        
    return recall, precision, f1

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script using argparse')
    
    # Add the arguments
    parser.add_argument('--in_file', type=str, required=True, help='Input file')
    parser.add_argument('--out_file', type=str, required=True, help='Out file')
    
    # Parse the arguments
    args = parser.parse_args()
    
    
    df = pd.read_csv(args.in_file)
    
    df['Similarity'] = 1 / (df['Similarity'] + 1.0)
        
    # edges = df[['D1', 'D2', 'Similarity']].values.tolist()
    edges = [[int(row['D1']), int(row['D2']), row['Similarity']] for i, row in df.iterrows()]
    edges = sorted(edges, key=lambda x: x[2], reverse=True)
    ground_results = set([(int(row['D1']), int(row['True'])) for i, row in df.iterrows()])
    
    min_collection = df.D1.unique().shape[0]
    l_matched = set()
    r_matched = set()
        
    matching_time = time()
    
    results = []
    for noind, (i, j, dist) in enumerate(edges):
    
        if i in l_matched or j in r_matched:
            continue
        results.append((i, j))
        l_matched.add(i)
        r_matched.add(j)
    
        if len(results) == min_collection:
            break
            
    results = pd.DataFrame(results, columns=['D1', 'D2'])

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    results.to_csv(args.out_file, header=True, index=False)