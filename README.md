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
### Annotate

To run LLM - Random & Sampled (Q1):
```
cd scripts/annotate/
./run_llm_noisy_data_selection.sh
```

To run LLM - Ground (Q2):
```
cd scripts/annotate/
./run_llm_ground.sh
```

To run LLM - Noisy (Q2):
```
cd scripts/annotate/
./run_llm_noisy.sh
```

To run Hybrid - Ground (Q2):
```
cd scripts/annotate/
./run_hybrid_ground.sh
```

To run Hybrid - Noisy (Q2):
```
cd scripts/annotate/
./run_hybrid_noisy.sh
```

- Blocking: [Blocking & Sampling](scripts/blocking/README.md)
- Baselines: [Experiments](scripts/matching/baselines/README.md)
- Label Refinement: [Experiments](scripts/matching/label_refinement/README.md) [Evaluations](scripts/evaluate/label_refinement/README.md)
- PLM: [Experiments](scripts/matching/plm/README.md) [Evaluations](scripts/plm/voting/README.md)
- LLM: [Experiments](scripts/matching/llm/README.md) [Evaluations](scripts/evaluate/llm/README.md)

## Models
You can find the fine-tuned version of Mistral [here](https://huggingface.co/alexZeakis/AvengER-mistral-v0.3/) and the fine-tuned version of Llama [here](https://huggingface.co/alexZeakis/AvengER-llama-3.1/).


