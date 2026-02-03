"""
Vanilla Knowledge Distillation (Hinton et al. 2015)
Trains student using soft targets from teacher with temperature scaling.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim


def distillation_loss(student_logits, teacher_soft_targets, hard_targets,
                      temperature, alpha=0.1):
    """
    Combined loss for knowledge distillation.

    Args:
        student_logits: Raw outputs from student model
        teacher_soft_targets: Soft probabilities from teacher (at temperature T)
        hard_targets: Ground truth labels
        temperature: Temperature for distillation
        alpha: Weight for hard target loss (1-alpha is weight for soft targets)

    Returns:
        Combined loss value
    """
    # Soft target loss: KL divergence between student and teacher (both at temperature T)
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    soft_loss = F.kl_div(student_soft, teacher_soft_targets, reduction='batchmean')

    # Scale by T^2 as per paper (gradients scale as 1/T^2)
    soft_loss = soft_loss * (temperature ** 2)

    # Hard target loss: Standard cross-entropy (at temperature 1)
    hard_loss = F.cross_entropy(student_logits, hard_targets)

    # Weighted combination
    return alpha * hard_loss + (1 - alpha) * soft_loss


def train_vanilla_kd(student, teacher, train_loader, temperature=20,
                     alpha=0.1, epochs=20, lr=0.001, device='cuda'):
    """
    Train student model using vanilla knowledge distillation.
    
    Args:
        student: Student model to train
        teacher: Pre-trained teacher model (frozen)
        train_loader: DataLoader for training data
        temperature: Temperature for soft targets (default: 20)
        alpha: Weight for hard loss (default: 0.1)
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Trained student model
    """
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()  # Freeze teacher
    
    optimizer = optim.Adam(student.parameters(), lr=lr)
    
    student.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Get teacher's soft targets at temperature T
            with torch.no_grad():
                teacher_soft_targets, _ = teacher(data, temperature=temperature)
            
            # Get student outputs
            _, student_logits = student(data)
            
            # Compute distillation loss
            loss = distillation_loss(student_logits, teacher_soft_targets,
                                    target, temperature, alpha)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Vanilla KD Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return student


if __name__ == "__main__":
    # Example usage
    from models.mnist_models import TeacherNet, StudentNet
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    print("Testing Vanilla KD implementation...")
    
    # Create dummy models
    teacher = TeacherNet()
    student = StudentNet()
    
    # Dummy data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    
    # Test training (1 epoch)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    student = train_vanilla_kd(student, teacher, train_loader, epochs=1, device=device)
    
    print("✓ Vanilla KD implementation tested successfully")
