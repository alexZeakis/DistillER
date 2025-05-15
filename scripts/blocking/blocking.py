import os
import pandas as pd
import torch
from time import time
import argparse
import json

def find_exact_nns(tensor1, tensor2, tensor1_index, tensor2_index, k, device="cuda:0"):
    device = torch.device(device)
    
    tensor11 = torch.Tensor(tensor1).to(device)
    tensor22 = torch.Tensor(tensor2).to(device)
    
    dists = torch.cdist(tensor11, tensor22, p=2)

    topk_dists = torch.topk(dists, k, largest=False)
    results = []
    for no in range(tensor11.size(0)):
        for i in range(k):
            index = topk_dists.indices[no, i].item()
            score = topk_dists.values[no, i].item()
            actual_no = int(tensor1_index[no])
            actual_index = int(tensor2_index[index])
            # if actual_index == 1063 or actual_no == 1063:
                # print(no, index, actual_no, actual_index)
            results.append((actual_no, actual_index, score))
    
    return results

order = {'D2': {'D1': 'buy', 'D2': 'abt'},
        'D3': {'D1': 'amazon', 'D2': 'gp'},
        'D4': {'D1': 'acm', 'D2': 'dblp'},
        'D5': {'D1': 'imdb', 'D2': 'tmdb'},
        'D6': {'D1': 'imdb', 'D2': 'tvdb'},
        'D7': {'D1': 'tvdb', 'D2': 'tmdb'},
        'D8': {'D1': 'walmart', 'D2': 'amazon'},
        'D9': {'D1': 'dblp', 'D2': 'scholar'},
        }
    

def calc_recall(true, preds):
    return len(true & preds) / len(true)

def calc_precision(true, preds):
    return len(true & preds) / len(preds)

def calc_f1(precision, recall):
    return 2 * precision * recall / (precision+recall)


if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description='Example script using argparse')
    
    # Add the arguments
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--logdir', type=str, required=True, help='Log Directory')
    parser.add_argument('--in_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--emb_dir', type=str, required=True, help='Embedding directory')
    parser.add_argument('--device', type=str, required=False, default='cuda', help='Device to run experiments')
    parser.add_argument('--k', type=int, required=True, default=10, help='Number of candidates per query entity')
    
    # Parse the arguments
    args = parser.parse_args()
    
    datasets = order[args.dataset]
    # swap = settings[dtype]['swap']
    # datasets = order_dataset[args.dataset]
    
    # Print the values of the arguments
    print(f'Dataset: {args.dataset}, k: {args.k}')

    # files = os.listdir(args.emb_dir+args.dataset)
    file1 = '{}{}/{}.csv'.format(args.emb_dir, args.dataset, datasets['D1'])
    file2 = '{}{}/{}.csv'.format(args.emb_dir, args.dataset, datasets['D2'])
    
    df1 = pd.read_csv(file1, header=None, index_col=0)
    df2 = pd.read_csv(file2, header=None, index_col=0)
    print(df1.shape, df2.shape)
    df1_index = df1.index
    df2_index = df2.index
    df1 = torch.Tensor(df1.values)
    df2 = torch.Tensor(df2.values)

        
    ground_file = [f for f in os.listdir(args.in_dir+args.dataset) if 'gt' in f][0]
    ground_file = '{}{}/{}'.format(args.in_dir, args.dataset, ground_file)
    ground_df = pd.read_csv(ground_file, sep=",")
    ground_results = set(ground_df.apply(lambda x: (x[0], x[1]), axis=1).values)
    # if args.dataset in swap:
    #     ground_results = set(ground_df.apply(lambda x: (x[1], x[0]), axis=1).values)
    # else:
    #     ground_results = set(ground_df.apply(lambda x: (x[0], x[1]), axis=1).values)
    
    #query2input
    q2i_time = time()
    q2i_results = find_exact_nns(df1, df2, df1_index, df2_index, args.k, args.device)
    q2i_time = time() - q2i_time
    q2i_score_results = set([(x,y) for (x,y,_) in q2i_results]) 
    q2i_recall = calc_recall(ground_results, q2i_score_results)
    q2i_precision = calc_precision(ground_results, q2i_score_results)    
    
    #input2query
    i2q_time = time()
    i2q_results = find_exact_nns(df2, df1, df2_index, df1_index, args.k, args.device) # reverse
    i2q_time = time() - i2q_time
    i2q_results = [(y,x,score) for (x,y,score) in i2q_results]  #reverse 
    i2q_score_results = set([(x,y) for (x,y,_) in i2q_results]) 
    i2q_recall = calc_recall(ground_results, i2q_score_results)
    i2q_precision = calc_precision(ground_results, i2q_score_results)
    
    #union
    union_time = time()
    results_1 = sorted(q2i_results, key=lambda x: (x[0], x[1]))
    results_2 = sorted(i2q_results, key=lambda x: (x[0], x[1]))
    i, j = 0, 0
    union_results = []
    while i < len(results_1) and j < len(results_2):
        if results_1[i][0] < results_2[j][0]:
            union_results.append(results_1[i])
            i += 1
        elif results_1[i][0] == results_2[j][0]:
            if results_1[i][1] < results_2[j][1]:
                union_results.append(results_1[i])
                i += 1
            elif results_1[i][1] == results_2[j][1]:
                union_results.append(results_1[i])
                i += 1
            else:
                union_results.append(results_2[j])
                j += 1
        else:
            union_results.append(results_2[j])
            j += 1            
    # union_results = list(set(i2q_results) | set(q2i_results))
    union_time = time() - union_time
    union_score_results = set([(x,y) for (x,y,_) in union_results]) 
    union_recall = calc_recall(ground_results, union_score_results)
    union_precision = calc_precision(ground_results, union_score_results)

    #intersection
    intersection_time = time()
    # intersection_results = list(set(i2q_results) & set(q2i_results))
    results_1 = sorted(q2i_results, key=lambda x: (x[0], x[1]))
    results_2 = sorted(i2q_results, key=lambda x: (x[0], x[1]))
    i, j = 0, 0
    intersection_results = []
    while i < len(results_1) and j < len(results_2):
        if results_1[i][0] < results_2[j][0]:
            i += 1
        elif results_1[i][0] == results_2[j][0]:
            if results_1[i][1] < results_2[j][1]:
                i += 1
            elif results_1[i][1] == results_2[j][1]:
                intersection_results.append(results_1[i])
                i += 1
            else:
                j += 1
        else:
            j += 1        
    intersection_time = time() - intersection_time
    intersection_score_results = set([(x,y) for (x,y,_) in intersection_results]) 
    intersection_recall = calc_recall(ground_results, intersection_score_results)
    intersection_precision = calc_precision(ground_results, intersection_score_results)    
    
    # ground_dict = {k:v for (k,v) in ground_results}
    ground_results = [[int(x),int(y)] for x,y in ground_results]
    
    log = {'dataset': args.dataset, 'true_len': len(ground_results),
           'q2i_precision': q2i_precision, 'q2i_recall': q2i_recall, 'q2i_time':q2i_time, 'q2i_len': len(q2i_results),
           'i2q_precision': i2q_precision, 'i2q_recall': i2q_recall, 'i2q_time':i2q_time, 'i2q_len': len(i2q_results),
           'union_precision': union_precision, 'union_recall': union_recall, 'union_time':union_time, 'union_len': len(union_results),
           'intersection_precision': intersection_precision, 'intersection_recall': intersection_recall, 'intersection_time':intersection_time, 'intersection_len': len(intersection_results),
            'q2i_results': q2i_results, 'i2q_results': i2q_results,
            'union_results': union_results, 'intersection_results': intersection_results,
            'true': ground_results
           }
    
    path2 = args.logdir+"blocking_"+args.dataset+".json"
    os.makedirs(os.path.dirname(path2), exist_ok=True)
    with open(path2, 'w') as f:
        f.write(json.dumps(log, indent=4)+"\n")