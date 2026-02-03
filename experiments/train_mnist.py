"""
Main experiment runner for MNIST baseline experiments.
Trains teacher, normal student, and all distillation methods.
"""

import torch
import argparse
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import models
from models.mnist_models import (
    TeacherNet, StudentNet, FitNetStudent, FitNetRegressor, count_parameters
)

# Import training functions
from utils.training import train_teacher, train_student_normal, save_checkpoint
from distillation.vanilla_kd import train_vanilla_kd
from distillation.fitnet import train_fitnet_full
from distillation.rkd import train_rkd

# Import evaluation functions
from utils.evaluation import (
    evaluate, benchmark_inference, print_evaluation_summary
)


def get_mnist_dataloaders(batch_size=256, data_dir='./data'):
    """
    Create MNIST dataloaders with data augmentation.
    """
    # Data augmentation: random translation up to 2 pixels
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(2/28, 2/28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create results directory
    results_dir = Path(args.results_dir) / 'mnist' / 'baseline'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=args.batch_size, data_dir=args.data_dir
    )
    
    # ========================================================================
    # TRAIN TEACHER
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING TEACHER MODEL")
    print("=" * 70)
    
    teacher = TeacherNet(dropout_rate=args.dropout)
    teacher = train_teacher(
        teacher, train_loader, epochs=args.epochs, lr=args.lr, device=device
    )
    
    # Evaluate and save
    teacher_acc, teacher_err = evaluate(teacher, test_loader, device)
    print(f"\nTeacher Test Results: {teacher_acc:.2f}% accuracy ({teacher_err} errors)")
    
    save_checkpoint(
        teacher, None, args.epochs, teacher_acc,
        results_dir / 'teacher.pth',
        dropout_rate=args.dropout
    )
    
    # ========================================================================
    # TRAIN NORMAL STUDENT (Baseline)
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING NORMAL STUDENT (Baseline)")
    print("=" * 70)
    
    student_normal = StudentNet()
    student_normal = train_student_normal(
        student_normal, train_loader, epochs=args.epochs, lr=args.lr, device=device
    )
    
    # Evaluate and save
    student_normal_acc, student_normal_err = evaluate(student_normal, test_loader, device)
    print(f"\nNormal Student Test Results: {student_normal_acc:.2f}% accuracy ({student_normal_err} errors)")
    
    save_checkpoint(
        student_normal, None, args.epochs, student_normal_acc,
        results_dir / 'student_normal.pth'
    )
    
    # ========================================================================
    # TRAIN VANILLA KD STUDENT
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING VANILLA KD STUDENT")
    print("=" * 70)
    
    student_kd = StudentNet()
    student_kd = train_vanilla_kd(
        student_kd, teacher, train_loader,
        temperature=args.temperature, alpha=args.alpha,
        epochs=args.epochs, lr=args.lr, device=device
    )
    
    # Evaluate and save
    student_kd_acc, student_kd_err = evaluate(student_kd, test_loader, device)
    print(f"\nVanilla KD Test Results: {student_kd_acc:.2f}% accuracy ({student_kd_err} errors)")
    
    save_checkpoint(
        student_kd, None, args.epochs, student_kd_acc,
        results_dir / 'student_vanilla_kd.pth',
        temperature=args.temperature, alpha=args.alpha
    )
    
    # ========================================================================
    # TRAIN FITNET STUDENT
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING FITNET STUDENT")
    print("=" * 70)
    
    fitnet_student = FitNetStudent()
    fitnet_regressor = FitNetRegressor(student_dim=300, teacher_dim=1200)
    
    fitnet_student, fitnet_regressor = train_fitnet_full(
        fitnet_student, teacher, fitnet_regressor, train_loader,
        stage1_epochs=args.fitnet_stage1_epochs,
        stage2_epochs=args.epochs,
        temperature=args.temperature, alpha=args.alpha,
        lr=args.lr, device=device
    )
    
    # Evaluate and save
    fitnet_acc, fitnet_err = evaluate(fitnet_student, test_loader, device)
    print(f"\nFitNet Test Results: {fitnet_acc:.2f}% accuracy ({fitnet_err} errors)")
    
    save_checkpoint(
        fitnet_student, None, args.epochs, fitnet_acc,
        results_dir / 'student_fitnet.pth',
        temperature=args.temperature, alpha=args.alpha
    )
    
    # ========================================================================
    # TRAIN RKD STUDENT
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING RKD STUDENT")
    print("=" * 70)
    
    student_rkd = StudentNet()
    student_rkd = train_rkd(
        student_rkd, teacher, train_loader,
        beta=args.rkd_beta, epochs=args.epochs, lr=args.lr, device=device
    )
    
    # Evaluate and save
    student_rkd_acc, student_rkd_err = evaluate(student_rkd, test_loader, device)
    print(f"\nRKD Test Results: {student_rkd_acc:.2f}% accuracy ({student_rkd_err} errors)")
    
    save_checkpoint(
        student_rkd, None, args.epochs, student_rkd_acc,
        results_dir / 'student_rkd.pth',
        beta=args.rkd_beta
    )
    
    # ========================================================================
    # BENCHMARK INFERENCE SPEED
    # ========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARKING INFERENCE SPEED")
    print("=" * 70)
    
    sample_input = next(iter(test_loader))[0][:1]
    
    models = {
        'Teacher': teacher,
        'Student Normal': student_normal,
        'Student Vanilla KD': student_kd,
        'FitNet': fitnet_student,
        'Student RKD': student_rkd
    }
    
    results = {}
    for name, model in models.items():
        mean_ms, std_ms = benchmark_inference(
            model, sample_input, num_samples=10000, num_trials=5, device=device
        )
        print(f"{name:25s}: {mean_ms:.3f} ± {std_ms:.3f} ms/sample")
        
        # Collect results
        acc, err = evaluate(model, test_loader, device)
        results[name] = {
            'errors': err,
            'accuracy': acc,
            'params': count_parameters(model),
            'latency_mean': mean_ms,
            'latency_std': std_ms
        }
    
    # ========================================================================
    # PRINT FINAL SUMMARY
    # ========================================================================
    print_evaluation_summary(results)
    
    # Save results to file
    import json
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ All results saved to {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST Knowledge Distillation Experiments')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory for MNIST dataset')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate for teacher')
    
    # Distillation parameters
    parser.add_argument('--temperature', type=float, default=20,
                       help='Temperature for knowledge distillation')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Weight for hard loss in KD')
    parser.add_argument('--rkd-beta', type=float, default=100,
                       help='Weight for RKD distance loss')
    parser.add_argument('--fitnet-stage1-epochs', type=int, default=10,
                       help='Epochs for FitNet Stage 1 (hint training)')
    
    args = parser.parse_args()
    main(args)
