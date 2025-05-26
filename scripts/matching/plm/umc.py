import pandas as pd
from time import time
import argparse
import os

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
    parser.add_argument('--input_file', type=str, required=True, help='Input file')
    parser.add_argument('--predictions_file', type=str, required=True, help='Predictions file')
    parser.add_argument('--out_file', type=str, required=True, help='Out file')
    
    # Parse the arguments
    args = parser.parse_args()
    
    predictions = pd.read_csv(args.predictions_file, index_col=0)
    test = pd.read_csv(args.input_file)
    sims = {(int(row['D1']), int(row['D2'])): float(row['Similarity']) for index, row in test.iterrows()}
    
    predictions = predictions.loc[predictions.predictions == 1]
    predictions['Similarity'] = pd.Series([sims[int(row['left_ID']), int(row['right_ID'])] for index, row in predictions.iterrows()])
    predictions['Similarity'] = 1 / (predictions['Similarity'] + 1.0)
    
    # #print(total.loc[total.predictions == 1].groupby('D1')['D1'].count().describe())
    
    edges = predictions[['left_ID', 'right_ID', 'Similarity']].values
    edges = sorted(edges, key=lambda x: x[2], reverse=True)
    ground_results = set([(int(row['D1']), int(row['True'])) for index, row in test.iterrows()])
    
    min_collection = predictions.left_ID.unique().shape[0]
    l_matched = set()
    r_matched = set()
        
    matching_time = time()
    
    delta = 0.95
        
    scores = []
    results = []
    for noind, (i, j, dist) in enumerate(edges):
        while dist < delta:
            results2 = set(results)
            matching_time2 = time() - matching_time
            
            recall, precision, f1 = evaluate(ground_results, results2)
            scores.append((recall, precision, f1, matching_time2, len(results2), delta)) 
            delta -= 0.05
    
        if i in l_matched or j in r_matched:
            continue
        results.append((i, j))
        l_matched.add(i)
        r_matched.add(j)
    
        if len(results) == min_collection:
            break
            
    results = set(results)
    matching_time = time() - matching_time
    recall, precision, f1 = evaluate(ground_results, results)
    scores.append((recall, precision, f1, matching_time, len(results), delta))
        
    scores = pd.DataFrame(scores, columns=['Recall', 'Precision', 'F1', 
                                             'Matching Time', '#Results', 'Delta'])
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    scores.to_csv(args.out_file, header=True, index=False)
    
    