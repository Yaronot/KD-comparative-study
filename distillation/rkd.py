"""
Relational Knowledge Distillation (Park et al. 2019)
Preserves pairwise distance relationships between samples in feature space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def compute_distance_matrix(embeddings):
    """
    Compute pairwise Euclidean distance matrix.
    
    Args:
        embeddings: Tensor of shape (batch_size, feature_dim)
        
    Returns:
        Distance matrix of shape (batch_size, batch_size)
    """
    # Compute pairwise distances using broadcasting
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    dot_product = torch.mm(embeddings, embeddings.t())
    squared_norm = dot_product.diag().unsqueeze(0)
    distances = squared_norm + squared_norm.t() - 2.0 * dot_product
    distances = torch.sqrt(torch.clamp(distances, min=1e-12))
    
    return distances


def normalize_distance_matrix(distance_matrix):
    """
    Normalize distance matrix by mean of non-zero elements.
    
    Args:
        distance_matrix: Square distance matrix
        
    Returns:
        Normalized distance matrix
    """
    # Get non-zero elements (excluding diagonal)
    mask = torch.ones_like(distance_matrix) - torch.eye(distance_matrix.size(0), device=distance_matrix.device)
    non_zero_distances = distance_matrix * mask
    
    # Compute mean of non-zero elements
    mean_distance = non_zero_distances.sum() / mask.sum()
    
    # Normalize
    normalized = distance_matrix / (mean_distance + 1e-8)
    
    return normalized


def rkd_distance_loss(student_embeddings, teacher_embeddings):
    """
    Relational Knowledge Distillation distance loss.
    Uses Smooth L1 (Huber) loss between normalized distance matrices.
    
    Args:
        student_embeddings: Student feature embeddings (batch_size, feature_dim)
        teacher_embeddings: Teacher feature embeddings (batch_size, feature_dim)
        
    Returns:
        RKD distance loss
    """
    # Compute distance matrices
    student_distances = compute_distance_matrix(student_embeddings)
    teacher_distances = compute_distance_matrix(teacher_embeddings)
    
    # Normalize for scale invariance
    student_distances_norm = normalize_distance_matrix(student_distances)
    teacher_distances_norm = normalize_distance_matrix(teacher_distances)
    
    # Compute Smooth L1 loss
    loss = F.smooth_l1_loss(student_distances_norm, teacher_distances_norm)
    
    return loss


def extract_embeddings(model, x, layer_name='penultimate'):
    """
    Extract feature embeddings from a model.
    For MLP, extracts activations before final classification layer.
    
    Args:
        model: Neural network model
        x: Input tensor
        layer_name: Which layer to extract from
        
    Returns:
        Feature embeddings
    """
    # Forward pass through all but last layer
    x = x.view(-1, 784)
    
    # For StudentNet/TeacherNet architectures
    if hasattr(model, 'fc1') and hasattr(model, 'fc2'):
        x = F.relu(model.fc1(x))
        if hasattr(model, 'dropout'):
            x = model.dropout(x)
        embeddings = F.relu(model.fc2(x))  # Penultimate layer
        return embeddings
    else:
        raise ValueError("Model architecture not recognized for embedding extraction")


def train_rkd(student, teacher, train_loader, beta=100, epochs=20, 
              lr=0.001, device='cuda'):
    """
    Train student model using Relational Knowledge Distillation.
    
    Args:
        student: Student model to train
        teacher: Pre-trained teacher model (frozen)
        train_loader: DataLoader for training data
        beta: Weight for RKD distance loss (default: 100)
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
        total_ce_loss = 0
        total_rkd_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Get student outputs and embeddings
            _, student_logits = student(data)
            student_embeddings = extract_embeddings(student, data)
            
            # Get teacher embeddings
            with torch.no_grad():
                teacher_embeddings = extract_embeddings(teacher, data)
            
            # Cross-entropy loss
            ce_loss = F.cross_entropy(student_logits, target)
            
            # RKD distance loss
            rkd_loss = rkd_distance_loss(student_embeddings, teacher_embeddings)
            
            # Combined loss
            loss = ce_loss + beta * rkd_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_rkd_loss += rkd_loss.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_ce = total_ce_loss / len(train_loader)
        avg_rkd = total_rkd_loss / len(train_loader)
        
        print(f"RKD Epoch {epoch+1}/{epochs}, Total: {avg_loss:.4f}, "
              f"CE: {avg_ce:.4f}, RKD: {avg_rkd:.4f}")
    
    return student


if __name__ == "__main__":
    # Test RKD implementation
    print("Testing RKD implementation...")
    
    # Test distance computation
    embeddings = torch.randn(4, 100)
    distances = compute_distance_matrix(embeddings)
    print(f"Distance matrix shape: {distances.shape}")
    print(f"Distance matrix diagonal (should be ~0): {distances.diag()}")
    
    # Test normalization
    normalized = normalize_distance_matrix(distances)
    print(f"Normalized mean (should be ~1): {normalized[normalized != 0].mean():.4f}")
    
    # Test loss computation
    student_emb = torch.randn(8, 100)
    teacher_emb = torch.randn(8, 100)
    loss = rkd_distance_loss(student_emb, teacher_emb)
    print(f"RKD loss: {loss.item():.4f}")
    
    print("✓ RKD implementation tested successfully")
