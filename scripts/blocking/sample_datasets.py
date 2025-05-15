import pandas as pd
import json
import os
import argparse

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
    parser.add_argument('--in_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--existing_file', type=str, required=False, help='Existing file')
    parser.add_argument('--blocking_option', type=str, choices=['q2i', 'i2q', 'intersection', 'union'],
                                                              required=True, help='Blocking Option')
    parser.add_argument('--pos_number', type=int, default=300, required=False, help='Number of positive data')
    parser.add_argument('--neg_number', type=int, default=100, required=False, help='Number of negative data')
    
    # Parse the arguments
    args = parser.parse_args()
    
    with open(args.in_dir + "blocking_" + args.dataset + ".json") as f:
        j = json.load(f)
        
    ground = { k: v for k,v in j['true']}
    results = j[args.blocking_option + '_results']
    df = pd.DataFrame(results, columns=['D1','D2','Similarity'])
    df['True'] = df.D1.apply(lambda x: ground.get(x, -1))
    
    if args.existing_file is not None:
        df2 = pd.read_csv(args.existing_file)
        existing_ids = set(df2.D1.unique())
        df = df.loc[df.D1.apply(lambda x: x not in existing_ids)]
    
    classes = {}
    for index, row in df.iterrows():
        if row['D2'] == row['True']:
            classes[row['D1']] = True
        else:
            if row['D1'] not in classes or not classes[row['D1']]:
                classes[row['D1']] = False
                
    pos_ids, neg_ids = [], []
    for key, val in classes.items():
        if val:
            pos_ids.append(key)
        else:
            neg_ids.append(key)
    
    # pos_ids = set(df.D1.loc[df['True'] >= 0].drop_duplicates().sample(args.pos_number, random_state=1924).values)
    # neg_ids = df.D1.loc[df['True'] < 0].drop_duplicates()
    # neg_ids = set(neg_ids.sample(neg_number, random_state=1924).values)
    # total_ids = pos_ids | neg_ids

    sel_pos_ids = set(pd.Series(pos_ids).sample(args.pos_number, random_state=1924).values)
    neg_number = min(args.neg_number, len(neg_ids))
    sel_neg_ids = set(pd.Series(neg_ids).sample(neg_number, random_state=1924).values)
    total_ids = sel_pos_ids | sel_neg_ids
    
    df = df.loc[df.D1.apply(lambda x: x in total_ids)]
    
    path2 = args.out_dir + args.dataset + ".csv"
    os.makedirs(os.path.dirname(path2), exist_ok=True)
    df.to_csv(path2, header=True, index=False)
        
    preds = set(df.apply(lambda x: (int(x['D1']), int(x['D2'])), axis=1).values)
    true = set(df.loc[df['True'] != -1].apply(lambda x: (int(x['D1']), int(x['True'])), axis=1).values)
    
    precision = calc_precision(true, preds)
    recall = calc_recall(true, preds)
    f1 = calc_f1(precision, recall)
    
    log = {'precision': precision, 'recall': recall, 'f1': f1,
           'required_positive': args.pos_number, 'available_positive': len(pos_ids), 'actual_positive': len(sel_pos_ids), 
           'required_negative': args.neg_number, 'available_negative': len(neg_ids), 'actual_negative': len(sel_neg_ids), 
           'dataset': args.dataset, 'blocking_option': args.blocking_option}
    
    path2 = args.out_dir + 'sampling_stats.jsonl'
    os.makedirs(os.path.dirname(path2), exist_ok=True)    
    with open(path2, 'a') as f:
       f.write(json.dumps(log)+"\n")