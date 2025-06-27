import pandas as pd
import json
import os
import argparse

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Train/Test Split Script')

    # Add arguments
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--in_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to store train/test splits')
    parser.add_argument('--split_percent', type=float, default=0.1, help='Percentage of data for training (default: 80)')
    parser.add_argument('--blocking_option', type=str, choices=['q2i', 'i2q', 'intersection', 'union'],
                                                              required=True, help='Blocking Option')
    parser.add_argument('--method', type=str, choices=['random', 'blocking', 'sampled'],
                        default='blocking', help='Sampling method.')
    parser.add_argument('--positive_ratio', type=float, default=0.75, help='Ratio for positive, if method=blocking.')

    # Parse the arguments
    args = parser.parse_args()

    # Read input data
    with open(args.in_dir + "blocking_" + args.dataset + ".json") as f:
        j = json.load(f)
        
    ground = { k: v for k,v in j['true']}
    results = j[args.blocking_option + '_results']
    df = pd.DataFrame(results, columns=['D1','D2','Similarity'])
    df['True'] = df.D1.apply(lambda x: ground.get(x, -1))

    entities = df['D1'].drop_duplicates()
    print('\tTotal Entities: {:,}'.format(entities.shape[0]))
    n_samples = int(args.split_percent * entities.shape[0])
    print('\tSampled Entities: {:,}'.format(n_samples))
    
    if args.method == 'random':
        sampled_ids = set(entities.sample(n_samples, random_state=1924).values)
    elif args.method == 'blocking':
        df['Similarity'] = 1 / (df['Similarity'] + 1.0)
        sorted_df = df.groupby('D1')['Similarity'].max().sort_values(ascending=False)
        n_pos = int(args.positive_ratio * n_samples)
        n_neg = n_samples - n_pos
        positive_ids = sorted_df.head(n_pos).index.tolist()
        negative_ids = sorted_df.tail(n_neg).index.tolist()
        sampled_ids = set(positive_ids + negative_ids)
    elif args.method == 'sampled':
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
        
        n_pos = int(args.positive_ratio * n_samples)
        n_neg = n_samples - n_pos
        
        sel_pos_ids = set(pd.Series(pos_ids).sample(n_pos, random_state=1924).values)
        neg_number = min(n_neg, len(neg_ids))
        sel_neg_ids = set(pd.Series(neg_ids).sample(n_neg, random_state=1924).values)
        sampled_ids = sel_pos_ids | sel_neg_ids
        
    
    train_df = df.loc[df.D1.apply(lambda x: x in sampled_ids)]
    test_df = df.loc[df.D1.apply(lambda x: x not in sampled_ids)]

    # Ensure output directory exists
    os.makedirs(os.path.join(args.out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "test"), exist_ok=True)

    # Save splits
    train_path = os.path.join(args.out_dir, "train", f"{args.dataset}.csv")
    test_path = os.path.join(args.out_dir, "test", f"{args.dataset}.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Log stats
    stats = {
        "total_samples": len(df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "split_percent": args.split_percent,
        'total_entities': entities.shape[0],
        "sampled_entities": n_samples,
        'dataset': args.dataset
    }

    log_path = os.path.join(args.out_dir, "split_stats.json")
    with open(log_path, 'a') as f:
        f.write(json.dumps(stats)+"\n")
