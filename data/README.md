# Data Directory

This directory is used to store the MNIST and CIFAR-10 datasets.

## Automatic Download

The datasets will be automatically downloaded when you run the experiments for the first time:

- **MNIST**: ~60,000 training images, 10,000 test images (28×28 grayscale)
- **CIFAR-10**: ~50,000 training images, 10,000 test images (32×32 RGB)

## Manual Download (Optional)

If you prefer to download the datasets manually:

### MNIST
```python
from torchvision import datasets
datasets.MNIST('./data', train=True, download=True)
datasets.MNIST('./data', train=False, download=True)
```

### CIFAR-10
```python
from torchvision import datasets
datasets.CIFAR10('./data', train=True, download=True)
datasets.CIFAR10('./data', train=False, download=True)
```

## Storage Requirements

- MNIST: ~12 MB
- CIFAR-10: ~170 MB
- Total: ~182 MB

## Directory Structure (after download)

```
data/
├── MNIST/
│   └── raw/
│       ├── train-images-idx3-ubyte
│       ├── train-labels-idx1-ubyte
│       ├── t10k-images-idx3-ubyte
│       └── t10k-labels-idx1-ubyte
└── cifar-10-batches-py/
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    ├── test_batch
    └── batches.meta
```
