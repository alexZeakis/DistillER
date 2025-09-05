# DistillER
DistillER: Ensembling and Fine-Tuning LLMs for SELECT Prompts in Entity Resolution

# Instructions

## Environment
To create environment, run:
```
python3.10 -m venv DistillER_Env
source DistillER_Env/bin/activate
pip install -r requirements.txt
```

## Datasets
Datasets can be found [here](data/ccer/cleaned/).

## Experiments

### Blocking
To run Vectorization with S-GTR-T5, Blocking and Sampling, run:

```
cd scripts/blocking/
./run_blocking.sh
```
### Data Selection (Q1)

To run LLM - Random & Sampled:
```
cd scripts/annotate/
./run_llm_noisy_data_selection.sh
```

### Annotate (Q2)
To run LLM - Ground:
```
cd scripts/annotate/
./run_llm_ground.sh
```

To run LLM - Noisy:
```
cd scripts/annotate/
./run_llm_noisy.sh
```

To run Hybrid - Noisy with different LLM Annotators:
```
cd scripts/annotate/
./run_hybrid_noisy_llm.sh
```

To run Hybrid - Noisy with different Training Size:
```
cd scripts/annotate/
./run_hybrid_noisy_size.sh
```

To run Hybrid - Ground:
```
cd scripts/annotate/
./run_hybrid_ground.sh
```

### Fine-Tuning (Q3)

To run LLM - Fine-Tuning:
```
cd scripts/llm/
./run_finetuning.sh
```

To run SLM - Fine-Tuning:
```
cd scripts/slm/
./run_data.sh
./run_finetuning.sh
```

### Disambiguation (Q4)

To run UMC:
```
cd scripts/disambiguation/
./run_umc.sh
```

To run SELECT:
```
cd scripts/disambiguation/
./run_select.sh
```

To run Hybrid:
```
cd scripts/slm/
./run_hybrid.sh
```


### SotA (Q5)

#### ZeroER
```
cd scripts/matching/sota/ZeroER
conda env create -f environment.yml
conda activate ZeroER

./run.sh 
```

#### CollaborEM
#python data/make_collaborem.py
```
gdown --fuzzy https://drive.google.com/file/d/1MHRfyk5bp7jv1dz-dCByhnTL483G43tR/view?usp=sharing
pip install conda-pack
mkdir -p er
tar -xzf er.tar.gz -C er
source er/bin/activate
gdown --fuzzy https://drive.google.com/file/d/13uzWfiZNfJewEkCtAfS4J1rX5td8CaWf/view?usp=sharing
unzip lm_model.zip

./run.sh
```



#### HierGAT
```
python -m venv HierGAT
source HierGAT/bin/activate
pip install -r requirements.txt

./run.sh 
```

#### Unicorn
```
python -m venv unicorn_env
source unicorn_env/bin/activate
pip install -r requirements.txt
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cu110

./run.sh
```

#### SudoWoodo
```
python -m venv sudowoodo_env
source sudowoodo_env/bin/activate
pip install -r requirements.txt
cd apex
python setup.py install
cd ../

./run.sh
```

- Blocking: [Blocking & Sampling](scripts/blocking/README.md)
- Baselines: [Experiments](scripts/matching/baselines/README.md)
- Label Refinement: [Experiments](scripts/matching/label_refinement/README.md) [Evaluations](scripts/evaluate/label_refinement/README.md)
- PLM: [Experiments](scripts/matching/plm/README.md) [Evaluations](scripts/plm/voting/README.md)
- LLM: [Experiments](scripts/matching/llm/README.md) [Evaluations](scripts/evaluate/llm/README.md)

## Models
You can find the fine-tuned version of Mistral [here](https://huggingface.co/alexZeakis/AvengER-mistral-v0.3/) and the fine-tuned version of Llama [here](https://huggingface.co/alexZeakis/AvengER-llama-3.1/).


