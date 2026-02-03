"""
Model evaluation utilities: accuracy, errors, inference time.
"""

import torch
import time
import numpy as np


def evaluate(model, test_loader, device='cuda'):
    """
    Evaluate model accuracy and error count.
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        
    Returns:
        accuracy (float): Accuracy percentage
        errors (int): Number of incorrect predictions
    """
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs, _ = model(data, temperature=1.0)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    errors = total - correct
    
    return accuracy, errors


def get_predictions(model, test_loader, device='cuda'):
    """
    Get all predictions from a model on test set.
    
    Args:
        model: Model to get predictions from
        test_loader: DataLoader for test data
        device: Device to run on
        
    Returns:
        predictions (np.array): Predicted labels
        labels (np.array): True labels
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            outputs, _ = model(data, temperature=1.0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def benchmark_inference(model, sample_input, num_samples=10000, num_trials=5, device='cuda'):
    """
    Benchmark inference speed of a model.
    
    Args:
        model: Model to benchmark
        sample_input: Single input sample for inference
        num_samples: Number of inference iterations per trial
        num_trials: Number of trials to average over
        device: Device to run on
        
    Returns:
        mean_ms (float): Mean inference time in milliseconds
        std_ms (float): Standard deviation of inference time
    """
    model = model.to(device)
    sample_input = sample_input.to(device)
    model.eval()
    
    times = []
    
    with torch.no_grad():
        # Warmup
        for _ in range(100):
            model(sample_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark trials
        for trial in range(num_trials):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            
            for _ in range(num_samples):
                model(sample_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000 / num_samples  # Convert to ms
            times.append(elapsed)
    
    mean_ms = np.mean(times)
    std_ms = np.std(times)
    
    return mean_ms, std_ms


def count_parameters(model):
    """
    Count total trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        num_params (int): Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_evaluation_summary(results_dict):
    """
    Print formatted evaluation summary table.
    
    Args:
        results_dict: Dictionary with keys as model names and values as dicts
                     containing 'accuracy', 'errors', 'params', 'latency_mean', 'latency_std'
    """
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Model':<25} {'Errors':<10} {'Accuracy':<12} {'Params':<15} {'Latency (ms)'}")
    print("-" * 80)
    
    for model_name, results in results_dict.items():
        errors = results['errors']
        accuracy = results['accuracy']
        params = results['params']
        
        if 'latency_mean' in results and 'latency_std' in results:
            latency_str = f"{results['latency_mean']:.3f} ± {results['latency_std']:.3f}"
        else:
            latency_str = "N/A"
        
        print(f"{model_name:<25} {errors:<10} {accuracy:>6.2f}%      {params:>10,}     {latency_str}")
    
    print("=" * 80)


if __name__ == "__main__":
    print("Testing evaluation utilities...")
    
    from models.mnist_models import TeacherNet, StudentNet
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    # Dummy data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test models
    teacher = TeacherNet().to(device)
    student = StudentNet().to(device)
    
    # Test evaluation
    print("\nTesting accuracy evaluation...")
    teacher_acc, teacher_err = evaluate(teacher, test_loader, device)
    print(f"Teacher: {teacher_acc:.2f}% accuracy ({teacher_err} errors)")
    
    # Test inference benchmarking
    print("\nTesting inference benchmark...")
    sample = next(iter(test_loader))[0][:1]
    mean_ms, std_ms = benchmark_inference(teacher, sample, num_samples=1000, num_trials=3, device=device)
    print(f"Inference time: {mean_ms:.3f} ± {std_ms:.3f} ms")
    
    # Test summary printing
    print("\nTesting summary printing...")
    results = {
        'Teacher': {
            'errors': teacher_err,
            'accuracy': teacher_acc,
            'params': count_parameters(teacher),
            'latency_mean': mean_ms,
            'latency_std': std_ms
        }
    }
    print_evaluation_summary(results)
    
    print("\n✓ Evaluation utilities tested successfully")
