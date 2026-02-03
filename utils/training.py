"""
Common training utilities for all experiments.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim


def train_teacher(model, train_loader, epochs=20, lr=0.001, device='cuda'):
    """
    Train a teacher model with standard cross-entropy loss.
    
    Args:
        model: Teacher model to train
        train_loader: DataLoader for training data
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Trained teacher model
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            _, logits = model(data)
            loss = F.cross_entropy(logits, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Teacher Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
    
    return model


def train_student_normal(model, train_loader, epochs=20, lr=0.001, device='cuda'):
    """
    Train a student model normally (baseline without distillation).
    
    Args:
        model: Student model to train
        train_loader: DataLoader for training data
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Trained student model
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            _, logits = model(data)
            loss = F.cross_entropy(logits, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Student (Normal) Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
    
    return model


def save_checkpoint(model, optimizer, epoch, accuracy, path, **kwargs):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        accuracy: Model accuracy
        path: Path to save checkpoint
        **kwargs: Additional metadata to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        **kwargs
    }
    torch.save(checkpoint, path)
    print(f"✓ Checkpoint saved to {path}")


def load_checkpoint(model, path, device='cuda'):
    """
    Load model from checkpoint.
    
    Args:
        model: Model instance to load weights into
        path: Path to checkpoint file
        device: Device to load model onto
        
    Returns:
        Model with loaded weights and checkpoint metadata
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✓ Loaded checkpoint from {path}")
    if 'accuracy' in checkpoint:
        print(f"  Accuracy: {checkpoint['accuracy']:.2f}%")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    
    return model, checkpoint


if __name__ == "__main__":
    print("Testing training utilities...")
    
    from models.mnist_models import TeacherNet, StudentNet
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    # Dummy data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=2)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test teacher training (1 epoch)
    print("\nTesting teacher training...")
    teacher = TeacherNet()
    teacher = train_teacher(teacher, train_loader, epochs=1, device=device)
    
    # Test checkpoint saving/loading
    print("\nTesting checkpoint save/load...")
    save_checkpoint(teacher, None, epoch=1, accuracy=95.0, 
                   path='/tmp/test_checkpoint.pth', dropout_rate=0.2)
    
    teacher_new = TeacherNet()
    teacher_new, metadata = load_checkpoint(teacher_new, '/tmp/test_checkpoint.pth', device=device)
    
    print("\n✓ Training utilities tested successfully")
