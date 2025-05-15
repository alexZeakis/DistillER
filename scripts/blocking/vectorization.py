#!/usr/bin/env python
import argparse

import os
import pandas as pd
from time import time
import json

from sentence_transformers import SentenceTransformer

def create_embeddings(text, vectorizer, output_path, 
                      output_index, b=500, device="cuda"):
      init_time = time()
      if vectorizer == 'smpnet':
          model = SentenceTransformer('all-mpnet-base-v2', device=device)
      elif vectorizer == 'sgtrt5':
          model = SentenceTransformer('gtr-t5-base', device=device)
      elif vectorizer == 'sdistilroberta':
          model = SentenceTransformer('all-distilroberta-v1', device=device)
      elif vectorizer == 'sminilm':
          model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
      init_time = time() - init_time
     
      vect_time = 0
      with open(output_path, 'w') as o:
          total = len(range(0, len(text), b))
          for i in range(0, len(text), b):
              print(f'\r\t {i//b}/{total}', end='')
              t1 = time()
              temp_text = text[i:i+b]
              temp_index = output_index[i:i+b]
              vectors = model.encode(temp_text)
              t2 = time()
              vect_time += t2-t1
         
              #flushing
              df = pd.DataFrame(vectors)
              df.index = temp_index
              df.to_csv(o, index=True, header=False)
           
      log = {}
      log['init_time'] = init_time   
      log['time'] = vect_time
      log['dimensions'] = vectors.shape[1]
     
      return log

if __name__ == '__main__':
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script using argparse')

    # Add the arguments
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--logfile', type=str, required=True, help='Path to the logfile')
    parser.add_argument('--in_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, required=False, default='cuda', help='Device to run experiments')

    # Parse the arguments
    args = parser.parse_args()

    # Print the values of the arguments
    print(f'Dataset: {args.dataset}, Model path: {args.model}')
    
    for file in os.listdir(args.in_dir + args.dataset+"/"):
        if 'gt' in file:
            continue
        path = args.in_dir + args.dataset+"/"+file
        df = pd.read_csv(path, sep=",", index_col=0)
        df = df.fillna('')
        print("\n", file, df.shape)
        
        data = df.apply(lambda x: ' '.join([str(y) for y in x]), axis=1) # Schema-Agnostic format
        text = data.tolist()
        
        path2 = args.out_dir + args.dataset+"/" + file
        # path2 = path2.replace('.csv', f"_{args.model}.csv")

        os.makedirs(os.path.dirname(path2), exist_ok=True)
        os.makedirs(os.path.dirname(args.logfile), exist_ok=True)
        
        log = create_embeddings(text, args.model, path2, df.index, 
                                device=args.device)
        log['dir'] = args.dataset
        log['file'] = file
        log['vectorizer'] = args.model
        with open(args.logfile, 'a') as f:
            f.write(json.dumps(log)+"\n")
