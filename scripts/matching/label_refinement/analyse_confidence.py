import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

path = '../../../log/matching/confidence/noisy/partial_noisy/'
data = []
conf = []
labels = []
for file in os.listdir(path):
    with open(path+file) as f:
        j = json.load(f)
        
    for response in j['prompts']:
        data.append((response['confidence'], (response['noise_answer'] == response['ground_answer'])*1))
        # conf.append(response['confidence'])
        # labels.append(response['noise_answer'] == response['ground_answer'])

df = pd.DataFrame(data, columns=['Confidence', 'Label'])

# ax = df.Confidence.hist()
# ax.set_title('Histogram of Confidence')
plt.show()

ax = df.Label.hist()
ax.set_title('Histogram of Labels')
plt.show()

ax = df.loc[df.Label==1].Confidence.hist(color='#1f77b4')
ax.set_title('Histogram of Confidence - Class 1')
plt.show()

ax = df.loc[df.Label==0].Confidence.hist(color='#ff7f0e')
ax.set_title('Histogram of Confidence - Class 0')
plt.show()


y_true = df.Label.values
deltas = {}
for delta in range(0, 105, 1):
    delta = delta / 100
    y_pred = (df.Confidence >= delta).values * 1
    deltas[delta] = f1_score(y_true, y_pred)
deltas = pd.Series(deltas)
ax = deltas.plot.line()
ax.set_title('Threshold of Confidence')
ax.set_xlabel('Threshold')
ax.set_ylabel('F1-Score')

y_pred = (df.Confidence >= 0.9).values * 1
deltas[delta] = f1_score(y_true, y_pred)
print(confusion_matrix(y_true, y_pred))