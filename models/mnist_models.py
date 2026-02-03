"""
MLP architectures for MNIST experiments.
Includes baseline and modified (BatchNorm + Residual) versions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherNet(nn.Module):
    """
    Large teacher network for MNIST.
    Architecture: 784 -> 1200 -> 1200 -> 10
    With dropout regularization.
    """
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 10)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, temperature=1.0):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        return F.softmax(logits / temperature, dim=1), logits


class StudentNet(nn.Module):
    """
    Standard student network for MNIST.
    Architecture: 784 -> 800 -> 800 -> 10
    Used for Vanilla KD and RKD experiments.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 10)

    def forward(self, x, temperature=1.0):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.softmax(logits / temperature, dim=1), logits


class FitNetStudent(nn.Module):
    """
    Thin-deep student network for FitNets.
    Architecture: 784 -> 300 -> 300 -> 300 -> 300 -> 10
    Second hidden layer serves as 'guided layer' for hint matching.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 300)  # Guided layer
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, 300)
        self.fc5 = nn.Linear(300, 10)

    def forward(self, x, temperature=1.0, return_hint=False):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        hint = F.relu(self.fc2(x))  # Second layer serves as hint
        x = F.relu(self.fc3(hint))
        x = F.relu(self.fc4(x))
        logits = self.fc5(x)
        
        if return_hint:
            return F.softmax(logits / temperature, dim=1), logits, hint
        return F.softmax(logits / temperature, dim=1), logits


class FitNetRegressor(nn.Module):
    """
    Regressor network for FitNets Stage 1.
    Maps student guided layer activations to teacher hint layer dimensions.
    """
    def __init__(self, student_dim=300, teacher_dim=1200):
        super().__init__()
        self.regressor = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_hint):
        return self.regressor(student_hint)


# ============================================================================
# Modified Architectures (BatchNorm + Residual Connections)
# ============================================================================

class StudentNetModified(nn.Module):
    """
    Student network with BatchNorm and residual connections.
    Architecture: 784 -> 800 -> 800 -> 10
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 800)
        self.bn1 = nn.BatchNorm1d(800)
        self.fc2 = nn.Linear(800, 800)
        self.bn2 = nn.BatchNorm1d(800)
        self.fc3 = nn.Linear(800, 10)

    def forward(self, x, temperature=1.0):
        x = x.view(-1, 784)
        
        # First layer with BatchNorm (no residual from input)
        x = F.relu(self.bn1(self.fc1(x)))
        
        # Second layer with BatchNorm + Residual
        identity = x
        x = self.bn2(self.fc2(x))
        x = F.relu(x + identity)
        
        logits = self.fc3(x)
        return F.softmax(logits / temperature, dim=1), logits


class FitNetStudentModified(nn.Module):
    """
    FitNet student with BatchNorm and residual connections.
    Architecture: 784 -> 300 -> 300 -> 300 -> 300 -> 10
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 300)
        self.bn1 = nn.BatchNorm1d(300)
        
        self.fc2 = nn.Linear(300, 300)  # Guided layer
        self.bn2 = nn.BatchNorm1d(300)
        
        self.fc3 = nn.Linear(300, 300)
        self.bn3 = nn.BatchNorm1d(300)
        
        self.fc4 = nn.Linear(300, 300)
        self.bn4 = nn.BatchNorm1d(300)
        
        self.fc5 = nn.Linear(300, 10)

    def forward(self, x, temperature=1.0, return_hint=False):
        x = x.view(-1, 784)
        
        # First layer
        x = F.relu(self.bn1(self.fc1(x)))
        
        # Second layer (guided) with residual
        identity = x
        hint = self.bn2(self.fc2(x))
        hint = F.relu(hint + identity)
        
        # Third layer with residual
        identity = hint
        x = self.bn3(self.fc3(hint))
        x = F.relu(x + identity)
        
        # Fourth layer with residual
        identity = x
        x = self.bn4(self.fc4(x))
        x = F.relu(x + identity)
        
        logits = self.fc5(x)
        
        if return_hint:
            return F.softmax(logits / temperature, dim=1), logits, hint
        return F.softmax(logits / temperature, dim=1), logits


def count_parameters(model):
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    print("=" * 60)
    print("MNIST Model Architectures")
    print("=" * 60)
    
    teacher = TeacherNet()
    student = StudentNet()
    fitnet = FitNetStudent()
    student_mod = StudentNetModified()
    fitnet_mod = FitNetStudentModified()
    
    print(f"Teacher:                  {count_parameters(teacher):,} parameters")
    print(f"Student (baseline):       {count_parameters(student):,} parameters")
    print(f"FitNet (baseline):        {count_parameters(fitnet):,} parameters")
    print(f"Student (modified):       {count_parameters(student_mod):,} parameters")
    print(f"FitNet (modified):        {count_parameters(fitnet_mod):,} parameters")
    print("=" * 60)
    
    # Test forward pass
    x = torch.randn(2, 1, 28, 28)
    _, logits = teacher(x)
    print(f"Output shape: {logits.shape}")
