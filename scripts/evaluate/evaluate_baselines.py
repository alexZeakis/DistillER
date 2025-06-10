import os
import pandas as pd
import json
import re
import matplotlib.pyplot as plt



    

order = [f'D{no}' for no in range(2,10)]



################### COMPARING LLMS ON TEST DATA #########################

total_df_test = pd.DataFrame()

total_df_test['UMC'] = prepare_umc_file('../../log/matching/baselines/umc/')
path = '../../log/matching/'
total_df_test['Pretrained'] = prepare_pt_file(path+'baselines/pretrained/')    
total_df_test['GT'] = prepare_ft_file(path+'llm/ground/test_responses/')
# total_df['Llama3:70b'] = prepare_ft_file(path+'llm/noisy/test_responses/')
total_df_test['Llama3.1:8b'] = prepare_ft_file(path+'llm/noisy_llama_8/test_responses/')
total_df_test['Llama3.1:70b'] = prepare_ft_file(path+'llm/noisy_llama_70/test_responses/')
total_df_test['Qwen2.5:7b'] = prepare_ft_file(path+'llm/noisy_qwen_7/test_responses/')
total_df_test['Qwen2.5:32b'] = prepare_ft_file(path+'llm/noisy_qwen_32/test_responses/')

total_df_test = total_df_test.loc[order]

# total_df[['UMC', 'Pretrained', 'GT', 'Llama3.1:8b', 'Qwen2.5:32b']].plot.line()
create_plot_comparison(total_df_test)

total_df_test.loc['Mean', :] = total_df_test.mean()
total_df_test = total_df_test.round(2)




################### COMPARING LLMS ON TRAIN DATA #########################

path = '../../log/matching/llm/'
total_df_train = pd.DataFrame()
# total_df_2['Llama3:70b'] = prepare_pt_file(path+'noisy/partial_responses/')
# total_df_2['Llama3.3:70b'] = prepare_pt_file(path+'noisy_2/partial_responses/')
# total_df_2['Llama3.1:8b(Groq)'] = prepare_pt_file(path+'noisy_5/partial_responses/')
total_df_train['Llama3.1:8b'] = prepare_pt_file(path+'noisy_llama_8/partial_responses/')
total_df_train['Llama3.1:70b'] = prepare_pt_file(path+'noisy_llama_70/partial_responses/')
total_df_train['Qwen2.5:7b'] = prepare_pt_file(path+'noisy_qwen_7/partial_responses/')
total_df_train['Qwen2.5:32b'] = prepare_pt_file(path+'noisy_qwen_32/partial_responses/')

total_df_train = total_df_train.loc[order]
total_df_train.loc['Mean', :] = total_df_train.mean()
total_df_train = total_df_train.round(2)




################### COMPARING PLMS ON TEST DATA ###################


path = '../../log/matching/plm/'
total_df_plm = pd.DataFrame()
total_df_plm['GT'] = prepare_plm_file(path+'ground/')
total_df_plm['Llama3.1:8b'] = prepare_plm_file(path+'noisy_llama_8/')
total_df_plm['Llama3.1:70b'] = prepare_plm_file(path+'noisy_llama_70/')
total_df_plm['Qwen2.5:7b'] = prepare_plm_file(path+'noisy_qwen_7/')
total_df_plm['Qwen2.5:32b'] = prepare_plm_file(path+'noisy_qwen_32/')
# total_df_plm['GT_DS2:1'] = prepare_plm_file(path+'ground_ds/')
# total_df_plm['Llama3.1:70b_DS2:1'] = prepare_plm_file(path+'noisy_llama_70_ds/')

total_df_plm = total_df_plm.loc[order]
total_df_plm.loc['Mean', :] = total_df_plm.mean()
total_df_plm = total_df_plm.round(2)





################### COMPARING DOWNSAMPLING ON PLMS ON TEST DATA ###############

path = '../../log/matching/plm/'
path2 = '../../log/matching/plm_downsampling/'
total_df_plm_ds = pd.DataFrame()
total_df_plm_ds['GT'] = prepare_plm_file(path+'ground/')
total_df_plm_ds['Llama3.1:70b'] = prepare_plm_file(path+'noisy_llama_70/')
total_df_plm_ds['GT_DS2:1'] = prepare_plm_file(path2+'ground/')
total_df_plm_ds['Llama3.1:70b_DS2:1'] = prepare_plm_file(path2+'noisy_llama_70/')

total_df_plm_ds = total_df_plm_ds.loc[order]
total_df_plm_ds.loc['Mean', :] = total_df_plm_ds.mean()
total_df_plm_ds = total_df_plm_ds.round(2)



################### COMPARING PLMS - GENERALIZED ON TEST DATA #################


path = '../../log/matching/plm_generalized/'
total_df_plm_gen = pd.DataFrame()
total_df_plm_gen['GT'] = prepare_plm_file(path+'ground/')
total_df_plm_gen['Llama3.1:8b'] = prepare_plm_file(path+'noisy_llama_8/')
total_df_plm_gen['Llama3.1:70b'] = prepare_plm_file(path+'noisy_llama_70/')
total_df_plm_gen['Qwen2.5:7b'] = prepare_plm_file(path+'noisy_qwen_7/')
total_df_plm_gen['Qwen2.5:32b'] = prepare_plm_file(path+'noisy_qwen_32/')

total_df_plm_gen = total_df_plm_gen.loc[order]
total_df_plm_gen.loc['Mean', :] = total_df_plm_gen.mean()
total_df_plm_gen = total_df_plm_gen.round(2)



################### COMPARING UMC ON PLMS ON TEST DATA ###############


path = '../../log/matching/plm/'
total_df_plm_umc = pd.DataFrame()
total_df_plm_umc['GT - wo/ UMC'] = prepare_plm_file(path+'ground/')
total_df_plm_umc['GT - w/ UMC'] = prepare_umc_file(path+'ground/umc/')
total_df_plm_umc['Llama3.1:70b - wo/ UMC'] = prepare_plm_file(path+'noisy_llama_70/')
total_df_plm_umc['Llama3.1:70b - w/ UMC'] = prepare_umc_file(path+'noisy_llama_70/umc/')

total_df_plm_umc = total_df_plm_umc.loc[order]
total_df_plm_umc.loc['Mean', :] = total_df_plm_umc.mean()
total_df_plm_umc = total_df_plm_umc.round(2)





################### COMPARING LABEL REFINEMENT ON TEST DATA ###################


path = '../../log/matching/'
total_df_conf_test = pd.DataFrame()
total_df_conf_test['Llama3.1:8b'] = prepare_ft_file(path+'llm/noisy_llama_8/test_responses/')
total_df_conf_test['Llama3.1:8b - LR_θ=0.9'] = prepare_ft_file(path+'confidence/noisy_llama_8/test_responses/')

total_df_conf_test = total_df_conf_test.loc[order]
total_df_conf_test.loc['Mean', :] = total_df_conf_test.mean()
total_df_conf_test = total_df_conf_test.round(2)






################### COMPARING LABEL REFINEMENT ON TRAIN DATA ###################

path = '../../log/matching/'
total_df_conf_train = pd.DataFrame()
total_df_conf_train['Llama3.1:8b'] = prepare_pt_file(path+'llm/noisy_llama_8/partial_responses/')
total_df_conf_train['Llama3.1:8b - LR_θ=0.9'] = prepare_pt_file(path+'confidence/noisy_llama_8/partial_responses/')

total_df_conf_train = total_df_conf_train.loc[order]
total_df_conf_train.loc['Mean', :] = total_df_conf_train.mean()
total_df_conf_train = total_df_conf_train.round(2)





################### COMPARING PLM-Annotator ON TEST DATA #########################

total_df_plm_annot_test = pd.DataFrame()

total_df_plm_annot_test['UMC'] = prepare_umc_file('../../log/matching/baselines/umc/')
path = '../../log/matching/'
total_df_plm_annot_test['Pretrained'] = prepare_pt_file(path+'baselines/pretrained/')    
total_df_plm_annot_test['GT'] = prepare_ft_file(path+'label_refinement_plm/ground/test_responses/')
total_df_plm_annot_test['Qwen2.5:7b'] = prepare_ft_file(path+'label_refinement_plm/noisy_qwen_7/test_responses/')

total_df_plm_annot_test = total_df_plm_annot_test.loc[order]
total_df_plm_annot_test.loc['Mean', :] = total_df_plm_annot_test.mean()
total_df_plm_annot_test = total_df_plm_annot_test.round(2)
