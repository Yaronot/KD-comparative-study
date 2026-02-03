#!/bin/bash
# Run all MNIST experiments to reproduce paper results

echo "=========================================="
echo "Running MNIST Experiments"
echo "=========================================="

# Set parameters from paper
EPOCHS=20
TEMPERATURE=20
ALPHA=0.1
RKD_BETA=100
DROPOUT=0.2
BATCH_SIZE=256
LR=0.001

# Run baseline experiments
echo ""
echo "Running baseline MNIST experiments..."
python experiments/train_mnist.py \
    --epochs $EPOCHS \
    --temperature $TEMPERATURE \
    --alpha $ALPHA \
    --rkd-beta $RKD_BETA \
    --dropout $DROPOUT \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --results-dir ./results

# Run modified architecture experiments (BatchNorm + Residual)
echo ""
echo "Running modified architecture experiments..."
python experiments/train_mnist_modified.py \
    --epochs $EPOCHS \
    --temperature $TEMPERATURE \
    --alpha $ALPHA \
    --rkd-beta $RKD_BETA \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --results-dir ./results

echo ""
echo "=========================================="
echo "MNIST experiments complete!"
echo "Results saved to ./results/mnist/"
echo "=========================================="
