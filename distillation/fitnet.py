"""
FitNets: Hints for Thin Deep Nets (Romero et al. 2014)
Two-stage training: Stage 1 trains regressor, Stage 2 trains full network with KD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def extract_teacher_hint(teacher, x):
    """
    Extract hint layer activations from teacher (first hidden layer).
    
    Args:
        teacher: Teacher model
        x: Input tensor
        
    Returns:
        Teacher hint activations
    """
    x = x.view(-1, 784)
    hint = F.relu(teacher.fc1(x))  # First hidden layer
    if hasattr(teacher, 'dropout'):
        hint = teacher.dropout(hint)
    return hint


def train_fitnet_stage1(student, teacher, regressor, train_loader, 
                        epochs=10, lr=0.001, device='cuda'):
    """
    FitNets Stage 1: Train regressor to map student guided layer to teacher hint.
    
    Args:
        student: Student model (frozen except for layers up to guided layer)
        teacher: Pre-trained teacher model (frozen)
        regressor: Regressor network to train
        train_loader: DataLoader for training data
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Trained regressor
    """
    student = student.to(device)
    teacher = teacher.to(device)
    regressor = regressor.to(device)
    
    # Freeze all models
    teacher.eval()
    student.eval()
    
    # Only train regressor
    for param in student.parameters():
        param.requires_grad = False
    for param in teacher.parameters():
        param.requires_grad = False
        
    optimizer = optim.Adam(regressor.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Get teacher hint
            with torch.no_grad():
                teacher_hint = extract_teacher_hint(teacher, data)
                
                # Get student guided layer activation
                x = data.view(-1, 784)
                x = F.relu(student.fc1(x))
                student_hint = F.relu(student.fc2(x))  # Guided layer
            
            # Pass student hint through regressor
            regressed_hint = regressor(student_hint)
            
            # MSE loss between regressed student hint and teacher hint
            loss = F.mse_loss(regressed_hint, teacher_hint)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"FitNet Stage 1 Epoch {epoch+1}/{epochs}, Hint Loss: {avg_loss:.4f}")
    
    # Re-enable gradients for student
    for param in student.parameters():
        param.requires_grad = True
        
    return regressor


def train_fitnet_stage2(student, teacher, train_loader, temperature=20,
                       alpha=0.1, epochs=20, lr=0.001, device='cuda'):
    """
    FitNets Stage 2: Train full student network with knowledge distillation.
    The hint-trained layers provide favorable initialization.
    
    Args:
        student: Student model (with hint layers pre-trained in Stage 1)
        teacher: Pre-trained teacher model (frozen)
        train_loader: DataLoader for training data
        temperature: Temperature for soft targets
        alpha: Weight for hard loss
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
            
            # Get teacher's soft targets
            with torch.no_grad():
                teacher_soft_targets, _ = teacher(data, temperature=temperature)
            
            # Get student outputs
            _, student_logits = student(data)
            
            # Distillation loss (same as vanilla KD)
            student_soft = F.log_softmax(student_logits / temperature, dim=1)
            soft_loss = F.kl_div(student_soft, teacher_soft_targets, reduction='batchmean')
            soft_loss = soft_loss * (temperature ** 2)
            
            hard_loss = F.cross_entropy(student_logits, target)
            
            loss = alpha * hard_loss + (1 - alpha) * soft_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"FitNet Stage 2 Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return student


def train_fitnet_full(student, teacher, regressor, train_loader,
                      stage1_epochs=10, stage2_epochs=20, temperature=20,
                      alpha=0.1, lr=0.001, device='cuda'):
    """
    Full FitNets training pipeline (both stages).
    
    Args:
        student: Student model to train
        teacher: Pre-trained teacher model
        regressor: Regressor network for Stage 1
        train_loader: DataLoader for training data
        stage1_epochs: Epochs for Stage 1 (hint training)
        stage2_epochs: Epochs for Stage 2 (KD training)
        temperature: Temperature for Stage 2 soft targets
        alpha: Weight for hard loss in Stage 2
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Trained student model and regressor
    """
    print("=" * 60)
    print("FitNets Two-Stage Training")
    print("=" * 60)
    
    # Stage 1: Train regressor
    print("\n[Stage 1] Training regressor for hint matching...")
    regressor = train_fitnet_stage1(student, teacher, regressor, train_loader,
                                    epochs=stage1_epochs, lr=lr, device=device)
    
    # Stage 2: Train full network with KD
    print("\n[Stage 2] Training student with knowledge distillation...")
    student = train_fitnet_stage2(student, teacher, train_loader,
                                  temperature=temperature, alpha=alpha,
                                  epochs=stage2_epochs, lr=lr, device=device)
    
    print("\n✓ FitNets training complete")
    print("=" * 60)
    
    return student, regressor


if __name__ == "__main__":
    from models.mnist_models import TeacherNet, FitNetStudent, FitNetRegressor
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    print("Testing FitNets implementation...")
    
    # Create models
    teacher = TeacherNet()
    student = FitNetStudent()
    regressor = FitNetRegressor(student_dim=300, teacher_dim=1200)
    
    # Dummy data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    
    # Test Stage 1 (1 epoch)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\nTesting Stage 1...")
    regressor = train_fitnet_stage1(student, teacher, regressor, train_loader, 
                                    epochs=1, device=device)
    
    print("\n✓ FitNets implementation tested successfully")
