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
- Blocking: [Blocking & Sampling](scripts/blocking/README.md)
- Baselines: [Experiments](scripts/matching/baselines/README.md)
- Label Refinement: [Experiments](scripts/matching/label_refinement/README.md) [Evaluations](scripts/evaluate/label_refinement/README.md)
- PLM: [Experiments](scripts/matching/plm/README.md) [Evaluations](scripts/plm/voting/README.md)
- LLM: [Experiments](scripts/matching/llm/README.md) [Evaluations](scripts/evaluate/llm/README.md)

## Models
You can find the fine-tuned version of Mistral [here](https://huggingface.co/alexZeakis/AvengER-mistral-v0.3/) and the fine-tuned version of Llama [here](https://huggingface.co/alexZeakis/AvengER-llama-3.1/).


