# A Comparative Study of Knowledge Distillation Methods

**By:** Yaron Otmazgin & Yael Levy  
**Course:** Advanced Computational Learning (52025)

## Overview

This repository contains code for our systematic comparison of three prominent knowledge distillation methods:
- **Vanilla KD** (Hinton et al. 2015): Soft target matching with temperature scaling
- **FitNets** (Romero et al. 2014): Two-stage training with intermediate layer hints
- **RKD** (Park et al. 2019): Preserving pairwise relational structure

We evaluate these methods on MNIST (MLP architectures) and CIFAR-10 (CNN architectures), examining accuracy-compression-speed trade-offs.

## Key Results

### MNIST Results (MLP)

| Model | Errors | Accuracy | Params | Latency (ms) |
|-------|--------|----------|--------|--------------|
| Teacher | 105 | 98.95% | 2.4M | 0.216 ± 0.062 |
| Student Normal | 123 | 98.77% | 1.3M | 0.167 ± 0.020 |
| Vanilla KD | 115 | 98.85% | 1.3M | 0.174 ± 0.023 |
| FitNet | 117 | 98.83% | 509K | 0.263 ± 0.027 |
| RKD | 118 | 98.82% | 1.3M | 0.154 ± 0.002 |

### CIFAR-10 Results (CNN)

| Model | Errors | Accuracy | Params | Latency (ms) |
|-------|--------|----------|--------|--------------|
| Teacher | 1746 | 82.54% | 4.6M | 0.445 |
| Student Normal | 2191 | 78.09% | 1.1M | 0.344 |
| Vanilla KD | 2153 | 78.47% | 1.1M | 0.317 |
| FitNet | 2111 | 78.89% | 83K | 0.347 |
| RKD | 2101 | 78.99% | 1.1M | 0.311 |

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- numpy
- scipy
- matplotlib

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/KD-comparative-study.git
cd KD-comparative-study

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run MNIST Experiments

```bash
# Train all models on MNIST (baseline)
python experiments/train_mnist.py --epochs 20 --temperature 20

# Train with modified architecture (BatchNorm + Residual)
python experiments/train_mnist_modified.py --epochs 20
```

### Run CIFAR-10 Experiments

```bash
# Train all models on CIFAR-10
python experiments/train_cifar10.py --epochs 20 --temperature 4
```

### Arguments

Common arguments for all experiment scripts:

- `--data-dir`: Directory for dataset (default: `./data`)
- `--results-dir`: Directory to save results (default: `./results`)
- `--batch-size`: Batch size for training (default: 256 for MNIST, 128 for CIFAR-10)
- `--epochs`: Number of training epochs (default: 20)
- `--lr`: Learning rate (default: 0.001)
- `--temperature`: Temperature for KD (default: 20 for MNIST, 4 for CIFAR-10)
- `--alpha`: Weight for hard loss in KD (default: 0.1)
- `--rkd-beta`: Weight for RKD distance loss (default: 100)

## Reproduce Paper Results

To reproduce all results from the paper:

```bash
# Run all experiments
bash scripts/run_all_mnist.sh
bash scripts/run_all_cifar10.sh

# Generate figures
python scripts/generate_figures.py
```

This will create:
- Model checkpoints in `results/`
- Performance metrics in JSON format
- Plots in `figures/`

## Using Individual Distillation Methods

Each distillation method can be used independently:

### Vanilla KD

```python
from models.mnist_models import TeacherNet, StudentNet
from distillation.vanilla_kd import train_vanilla_kd

teacher = TeacherNet()  # Pre-trained
student = StudentNet()

student = train_vanilla_kd(
    student, teacher, train_loader,
    temperature=20, alpha=0.1, epochs=20
)
```

### FitNets

```python
from models.mnist_models import FitNetStudent, FitNetRegressor
from distillation.fitnet import train_fitnet_full

student = FitNetStudent()
regressor = FitNetRegressor(student_dim=300, teacher_dim=1200)

student, regressor = train_fitnet_full(
    student, teacher, regressor, train_loader,
    stage1_epochs=10, stage2_epochs=20
)
```

### RKD

```python
from models.mnist_models import StudentNet
from distillation.rkd import train_rkd

student = StudentNet()

student = train_rkd(
    student, teacher, train_loader,
    beta=100, epochs=20
)
```

## Evaluation

```python
from utils.evaluation import evaluate, benchmark_inference

# Evaluate accuracy
accuracy, errors = evaluate(model, test_loader)

# Benchmark inference speed
mean_ms, std_ms = benchmark_inference(model, sample_input)
```

## Statistical Testing

McNemar's test to compare model predictions:

```python
from utils.mcnemar import mcnemar_test, print_mcnemar_results
from utils.evaluation import get_predictions

preds_before, labels = get_predictions(model_before, test_loader)
preds_after, _ = get_predictions(model_after, test_loader)

results = mcnemar_test(preds_before, preds_after, labels)
print_mcnemar_results(results, "Before", "After")
```


## References

1. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*.

2. Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y. (2014). FitNets: Hints for thin deep nets. *arXiv preprint arXiv:1412.6550*.

3. Park, W., Kim, D., Lu, Y., & Cho, M. (2019). Relational knowledge distillation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 3967-3976).

## License

MIT License - see LICENSE file for details.

