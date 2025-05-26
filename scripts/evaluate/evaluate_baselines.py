import os
import pandas as pd
import json
import re
import matplotlib.pyplot as plt


def prepare_plm_file(path):
    scores = {}
    with open(path+'log/matching_supervised_dynamic.txt') as f:
        for line in f:
            j = json.loads(line)
            scores[j['data_name']] = j['f1']
    return pd.Series(scores)
        
def get_scores_from_ft_json(path):
    j = json.load(open(path))
    dataset = j['settings']['dataset']
    
    ground_results = set()
    predictions = set()
    for res in j['responses']:
        # print(res)

        answer = find_integer(res['response'])
        if len(answer) > 0:
            answer = answer['answer']
        
        ground_results.add((res['query_id'], res['answer']))
        if len(answer) > 0:
            try:
                answer = int(answer[1:-1])
            except:
                    answer = 0
            predictions.add((res['query_id'], answer))
        # if answer != res['answer']:
        #     print("Response: {}\nShortResponse: {}\nAnswer: {}\n".format(res['response'], answer, res['answer']))
            
    f1 = calc_f1(ground_results, predictions)
    
    # return ground_results, predictions
    return dataset, f1

def get_scores_from_json(path):
    j = json.load(open(path))
    dataset = j['settings']['dataset']
    
    ground_results = set()
    predictions = set()
    for res in j['responses']:
        # print(res)
        # if res['query_id'] == 2054:
        #     # print(res['answer'], res['explanation'])
        #     print(res)

        if res['explanation'] is None: #error
            if res['answer'] is None: #timeout
                answer = {}
            else: #error in Structured output
                answer = find_json(res['answer'])
                if len(answer) > 0:
                    answer = answer['answer']
        else:
            answer = res['answer']
        # if res['ground_answer'] == -1: #TODO: Check for certainty
        #     ground_answer = 0
        
        ground_results.add((res['query_id'], res['ground_answer']))
        if len(answer) > 0:
            try:
                answer = int(answer[1:-1])
            except:
                    answer = 0
            predictions.add((res['query_id'], answer))
            
    f1 = calc_f1(ground_results, predictions)
    
    # return ground_results, predictions
    return dataset, f1

def find_integer(text):
    pattern = r'\[\d+\]'
    matches = re.findall(pattern, text, re.DOTALL)
    last_match = matches[0] if matches else None
    if last_match is not None: #successful structured output
        return {'answer': last_match, 'explanation': ''}
    return {}

def find_json(text):
    # print(text)
    
    # pattern = r'({"answer":.*?"explanation".*?})'
    pattern = '\{\s*"answer"\s*:\s*".*?"\s*,\s*"explanation"\s*:\s*".*?"\s*\}'
    matches = re.findall(pattern, text, re.DOTALL)
    last_match = matches[-1] if matches else None
    if last_match is not None: #successful structured output
        try:
            return json.loads(last_match)
        except:
            pattern = '\"answer"\s*:\s*".*?"\s*,'
            matches = re.findall(pattern, text, re.DOTALL)
            last_match = matches[-1] if matches else None
            if last_match is not None: #successful structured output
                return {'answer': last_match, 'explanation': ''}
            
    pattern = r'\[\d+\]'
    matches = re.findall(pattern, text, re.DOTALL)
    last_match = matches[-1] if matches else None
    if last_match is not None: #successful structured output
        return {'answer': last_match, 'explanation': ''}
    return {}


def calc_f1(true, preds):
    if len(true) > 0:
        recall = len(true & preds) / len(true)
    if len(preds) > 0:
        precision = len(true & preds) / len(preds)
    if len(true) > 0:
        return 2 * precision * recall / (precision+recall)
    
    
def prepare_pt_file(path):
    temp_scores = {}
    for file in os.listdir(path):
        if 'responses' not in file:
            continue
        dataset, f1 = get_scores_from_json(path+file)
        temp_scores[dataset] = f1
    return pd.Series(temp_scores)

def prepare_ft_file(path):
    temp_scores = {}
    for file in os.listdir(path):
        if 'responses' not in file:
            continue
        dataset, f1 = get_scores_from_ft_json(path+file)
        temp_scores[dataset] = f1
    return pd.Series(temp_scores)

def prepare_umc_file(path):
    temp_scores = {}
    for file in os.listdir(path):
        df = pd.read_csv(path+file)
        temp_scores[file.split('.')[0]] = df.iloc[-1]['F1']
    total_df_test['UMC'] = pd.Series(temp_scores)
    return pd.Series(temp_scores)

def create_plot_comparison(total_df_test):
    colors = {
        "UMC": "blue",
        "Pretrained": "orange",
        "GT": "green",
        "Llama3.1:70b": "red",
        "Llama3.1:8b": "purple",
        "Qwen2.5:32b": "brown"
    }

    df_reset = total_df_test.reset_index().rename(columns={'index': 'Dataset'})
    ax = None
    for column in colors.keys():
        ax = df_reset.plot.scatter(
            x='Dataset',
            y=column,
            label=column,
            ax=ax,
            color=colors.get(column, 'gray'),
            s=60  # marker size
        )
    ax.set_title("Scatter Plot of Methods per Dataset")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("F1")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    

order = [f'D{no}' for no in range(2,10)]



################### COMPARING LLMS ON TEST DATA #########################

total_df_test = pd.DataFrame()

total_df_test['UMC'] = prepare_umc_file('../../log/matching/baselines/umc/')
path = '../../log/matching/'
total_df_test['Pretrained'] = prepare_pt_file(path+'baselines/pretrained/')    
total_df_test['GT'] = prepare_ft_file(path+'finetuning/ground/test_responses/')
# total_df['Llama3:70b'] = prepare_ft_file(path+'finetuning/noisy/test_responses/')
total_df_test['Llama3.1:8b'] = prepare_ft_file(path+'finetuning/noisy_3/test_responses/')
total_df_test['Llama3.1:70b'] = prepare_ft_file(path+'finetuning/noisy_6/test_responses/')
total_df_test['Qwen2.5:7b'] = prepare_ft_file(path+'finetuning/noisy_7/test_responses/')
total_df_test['Qwen2.5:32b'] = prepare_ft_file(path+'finetuning/noisy_4/test_responses/')

total_df_test = total_df_test.loc[order]

# total_df[['UMC', 'Pretrained', 'GT', 'Llama3.1:8b', 'Qwen2.5:32b']].plot.line()
create_plot_comparison(total_df_test)

total_df_test.loc['Mean', :] = total_df_test.mean()
total_df_test = total_df_test.round(2)




################### COMPARING LLMS ON TRAIN DATA #########################

path = '../../log/matching/finetuning/'
total_df_train = pd.DataFrame()
# total_df_2['Llama3:70b'] = prepare_pt_file(path+'noisy/partial_responses/')
# total_df_2['Llama3.3:70b'] = prepare_pt_file(path+'noisy_2/partial_responses/')
# total_df_2['Llama3.1:8b(Groq)'] = prepare_pt_file(path+'noisy_5/partial_responses/')
total_df_train['Llama3.1:8b'] = prepare_pt_file(path+'noisy_3/partial_responses/')
total_df_train['Llama3.1:70b'] = prepare_pt_file(path+'noisy_6/partial_responses/')
total_df_train['Qwen2.5:7b'] = prepare_pt_file(path+'noisy_7/partial_responses/')
total_df_train['Qwen2.5:32b'] = prepare_pt_file(path+'noisy_4/partial_responses/')

total_df_train = total_df_train.loc[order]
total_df_train.loc['Mean', :] = total_df_train.mean()
total_df_train = total_df_train.round(2)




################### COMPARING PLMS ON TEST DATA ###################


path = '../../log/matching/plm/'
total_df_plm = pd.DataFrame()
total_df_plm['GT'] = prepare_plm_file(path+'ground/')
total_df_plm['Llama3.1:8b'] = prepare_plm_file(path+'noisy_3/')
total_df_plm['Llama3.1:70b'] = prepare_plm_file(path+'noisy_6/')
total_df_plm['Qwen2.5:7b'] = prepare_plm_file(path+'noisy_7/')
total_df_plm['Qwen2.5:32b'] = prepare_plm_file(path+'noisy_4/')
# total_df_plm['GT_DS2:1'] = prepare_plm_file(path+'ground_ds/')
# total_df_plm['Llama3.1:70b_DS2:1'] = prepare_plm_file(path+'noisy_6_ds/')

total_df_plm = total_df_plm.loc[order]
total_df_plm.loc['Mean', :] = total_df_plm.mean()
total_df_plm = total_df_plm.round(2)





################### COMPARING DOWNSAMPLING ON PLMS ON TEST DATA ###############

path = '../../log/matching/plm/'
total_df_plm_ds = pd.DataFrame()
total_df_plm_ds['GT'] = prepare_plm_file(path+'ground/')
total_df_plm_ds['Llama3.1:70b'] = prepare_plm_file(path+'noisy_6/')
total_df_plm_ds['GT_DS2:1'] = prepare_plm_file(path+'ground_ds/')
total_df_plm_ds['Llama3.1:70b_DS2:1'] = prepare_plm_file(path+'noisy_6_ds/')

total_df_plm_ds = total_df_plm_ds.loc[order]
total_df_plm_ds.loc['Mean', :] = total_df_plm_ds.mean()
total_df_plm_ds = total_df_plm_ds.round(2)


################### COMPARING UMC ON PLMS ON TEST DATA ###############


path = '../../log/matching/plm/'
total_df_plm_umc = pd.DataFrame()
total_df_plm_umc['GT - wo/ UMC'] = prepare_plm_file(path+'ground/')
total_df_plm_umc['GT - w/ UMC'] = prepare_umc_file(path+'ground/umc/')
total_df_plm_umc['Llama3.1:70b - wo/ UMC'] = prepare_plm_file(path+'noisy_6/')
total_df_plm_umc['Llama3.1:70b - w/ UMC'] = prepare_umc_file(path+'noisy_6/umc/')

total_df_plm_umc = total_df_plm_umc.loc[order]
total_df_plm_umc.loc['Mean', :] = total_df_plm_umc.mean()
total_df_plm_umc = total_df_plm_umc.round(2)





################### COMPARING LABEL REFINEMENT ON TEST DATA ###################


path = '../../log/matching/'
total_df_conf_test = pd.DataFrame()
total_df_conf_test['Llama3.1:8b'] = prepare_ft_file(path+'finetuning/noisy_3/test_responses/')
total_df_conf_test['Llama3.1:8b - LR_θ=0.9'] = prepare_ft_file(path+'confidence/noisy/test_responses/')

total_df_conf_test = total_df_conf_test.loc[order]
total_df_conf_test.loc['Mean', :] = total_df_conf_test.mean()
total_df_conf_test = total_df_conf_test.round(2)






################### COMPARING LABEL REFINEMENT ON TRAIN DATA ###################

path = '../../log/matching/'
total_df_conf_train = pd.DataFrame()
total_df_conf_train['Llama3.1:8b'] = prepare_pt_file(path+'finetuning/noisy_3/partial_responses/')
total_df_conf_train['Llama3.1:8b - LR_θ=0.9'] = prepare_pt_file(path+'confidence/noisy/partial_responses/')

total_df_conf_train = total_df_conf_train.loc[order]
total_df_conf_train.loc['Mean', :] = total_df_conf_train.mean()
total_df_conf_train = total_df_conf_train.round(2)