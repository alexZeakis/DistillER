import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

def robust_score(group):
    sims = group['Similarity'].sort_values()
    q1, q3 = sims.quantile([0.25, 0.75])
    iqr = q3 - q1
    trimmed = sims[(sims > q1) & (sims < q3)]
    trimmed_mean = trimmed.mean() if len(trimmed) > 0 else sims.mean()
    return pd.Series({'robust_score': trimmed_mean - iqr})

def interquartile_mean(series):
    n = len(series)
    if n <= 2:
        # Fallback: just return mean for small samples
        return series.mean()
    
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    central_values = series[(series >= q1) & (series <= q3)]
    
    # Fallback again if central_values is empty (can happen with tied values)
    if len(central_values) == 0:
        return series.mean()
    
    return central_values.mean()

path = '../../../log/matching/confidence/noisy/partial_noisy/'
path2 = '../../../data/ccer/cleaned/fine_tuning/train/'
data = []
conf = []
labels = []
for file in os.listdir(path):
    with open(path+file) as f:
        j = json.load(f)
        
    edges = pd.read_csv(path2 + j['settings']['dataset'] + '.csv')
    agg = edges.groupby('D1')['Similarity'].agg(['mean', 'std', 'median'])
    agg['std'] = agg['std'].fillna(0)
    agg['cv_inverse'] = agg['mean'] / agg['std'].replace(0, 1e-8)
    
    min_cv = agg['cv_inverse'].min()
    max_cv = agg['cv_inverse'].max()
    agg['cv_inverse_norm'] = (agg['cv_inverse'] - min_cv) / (max_cv - min_cv + 1e-8)
    
    agg['adjusted_score'] = agg['mean'] - agg['std']
    
    # agg['robust_score'] = edges.groupby('D1', group_keys=False)[['Similarity']].apply(robust_score)
    agg['IQM'] = edges.groupby('D1')['Similarity'].apply(interquartile_mean)
    agg = agg.sort_values('IQM', ascending=False)
    agg = agg['IQM'].to_dict()
        
    for response in j['prompts']:
        data.append((response['confidence'], 
                     (response['noise_answer'] == response['ground_answer'])*1,
                    agg[response['query_id']])
                    )
        # conf.append(response['confidence'])
        # labels.append(response['noise_answer'] == response['ground_answer'])


    

df = pd.DataFrame(data, columns=['Confidence', 'Label', 'Difficulty'])

# ax = df.Confidence.hist()
# ax.set_title('Histogram of Confidence')
plt.show()

ax = df.Label.hist()
ax.set_title('Histogram of Labels')
plt.show()

# ax = df.loc[df.Label==1].Confidence.hist(color='#1f77b4')
# ax.set_title('Histogram of Confidence - Class 1')
# plt.show()

# ax = df.loc[df.Label==0].Confidence.hist(color='#ff7f0e')
# ax.set_title('Histogram of Confidence - Class 0')
# plt.show()

# ax = df.loc[df.Label==1].Difficulty.hist(color='#1f77b4')
# ax.set_title('Histogram of Difficulty - Class 1')
# plt.show()

# ax = df.loc[df.Label==0].Difficulty.hist(color='#ff7f0e')
# ax.set_title('Histogram of Difficulty - Class 0')
# plt.show()


for alpha in range(0,11,5):
    alpha = alpha / 10
    df['Final'] = df.apply(lambda x: x['Confidence']*alpha + x['Difficulty']*(1-alpha), axis=1)
    
    ax = df.loc[df.Label==1].Final.hist(color='#1f77b4')
    ax.set_title('Histogram of Final - Class 1 - (a={})'.format(alpha))
    plt.show()
    
    ax = df.loc[df.Label==0].Final.hist(color='#ff7f0e')
    ax.set_title('Histogram of Final - Class 0 - (a={})'.format(alpha))
    plt.show()
    
    
    y_true = df.Label.values
    deltas = {}
    for delta in range(0, 105, 1):
        delta = delta / 100
        y_pred = (df.Final >= delta).values * 1
        deltas[delta] = f1_score(y_true, y_pred)
    deltas = pd.Series(deltas)
    ax = deltas.plot.line()
    ax.set_title('Threshold of Final - (a={})'.format(alpha))
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-Score')
    plt.show()
    
    y_pred = (df.Final >= 0.9).values * 1
    deltas[delta] = f1_score(y_true, y_pred)
    print(confusion_matrix(y_true, y_pred))