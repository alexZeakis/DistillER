# DistillER
DistillER: Knowledge Distillation in Entity Resolution with Large Language Models

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

To replicate results, please follow the order of the experiments:
i) Blocking ii) Data Selection iii) Knowledge Elicitation iv) Distillation Algorithms

| Component / Question | Experiments | Evaluations | Logs |
|----------------------|-------------|-------------|------|
| Blocking | [Blocking](scripts/blocking/README.md) | - | [Logs](log/blocking/) |
| **Data Selection (Q1)** | [Experiments](scripts/matching/data_selection/README.md) | [Evaluations](scripts/evaluate/data_selection/README.md) | [Logs](log/matching/data_selection/) |
| **Knowledge Elicitation** |  |  |
| ├─ Teachers (Q2) | [Experiments](scripts/matching/annotate/README.md) | [Evaluations](scripts/evaluate/annotate/README.md) | [Logs](log/matching/annotate/) |
| └─ Explanations (Q5) | [Experiments](scripts/matching/explanations/README.md) | [Evaluations](scripts/evaluate/explanations/README.md) | [Logs](log/matching/explanations/) |
| **Distillation Algorithms** |  |  |
| ├─Baselines | [Experiments](scripts/matching/baselines/README.md) | - | [Logs](log/matching/baselines/) |
| ├─ Supervised Fine-Tuning (Q3) | [Experiments](scripts/matching/sft/README.md) | [Evaluations](scripts/evaluate/sft/README.md) | [Logs](log/matching/sft/) |
| └─ Reinforcement Learning (Q4) | [Experiments](scripts/matching/rl/README.md) | [Evaluations](scripts/evaluate/rl/README.md) | [Logs](log/matching/rl/) |
| **SotA (Q6)** | [Experiments](scripts/matching/sota/README.md) | [Evaluations](scripts/evaluate/sota/README.md) | [Logs](log/matching/sota/) |

