import numpy as np
import json_tricks

# Load the inputs
inputs = json_tricks.load('inputs.json')

# Our implementations
def get_covariant_coordinates(B, X):
    """
    Find covariant coordinates of vectors X with respect to basis B.
    
    Covariant coordinates are the dot products of each vector with each basis vector.
    If B is a basis matrix where each row is a basis vector, and X is a matrix where
    each row is a vector, then the covariant coordinates are computed as X @ B.T
    (i.e., each vector dotted with each basis vector).
    
    Args:
        B: numpy array of shape (n_basis, n_dim) - basis vectors as rows
        X: numpy array of shape (n_vectors, n_dim) - vectors as rows
    
    Returns:
        numpy array of shape (n_vectors, n_basis) - covariant coordinates
    """
    # Covariant coordinates are X @ B.T
    # Each row of X is dotted with each row of B
    return X @ B.T

def reconstruct_vectors(B, C):
    B_expanded = np.expand_dims(B, axis=1)  
    C_expanded = np.expand_dims(C, axis=2)  
    products = B_expanded * C_expanded 
    res = np.sum(products, axis=0) 
    return res.T

def contravariant_to_covariant(B, C):
    """
    Convert contravariant coordinates to covariant coordinates.
    
    Given basis vectors B and contravariant coordinates C, we:
    1. First reconstruct the original vectors: V = C.T @ B
    2. Then compute covariant coordinates: V @ B.T
    
    This can be simplified to: C.T @ B @ B.T = C.T @ (B @ B.T)
    where B @ B.T is the Gram matrix of the basis vectors.
    
    Args:
        B: numpy array of shape (n_basis, n_dim) - basis vectors as rows
        C: numpy array of shape (n_basis, n_vectors) - contravariant coordinates
    
    Returns:
        numpy array of shape (n_vectors, n_basis) - covariant coordinates
    """
    # Method 2: Direct computation using Gram matrix
    gram_matrix = B @ B.T  # Gram matrix of basis vectors
    return C.T @ gram_matrix

# Test the implementations
print("Testing implementations...")

# Test task 1 (reconstruct_vectors)
print("\nTask 1 - Reconstruct vectors:")
for i, input_data in enumerate(inputs['task1']):
    result = reconstruct_vectors(**input_data)
    print(f"Test {i+1}: Shape {result.shape}")
    print(f"Result:\n{result}")

# Test task 2 (get_covariant_coordinates)
print("\nTask 2 - Get covariant coordinates:")
for i, input_data in enumerate(inputs['task2']):
    result = get_covariant_coordinates(**input_data)
    print(f"Test {i+1}: Shape {result.shape}")
    print(f"Result:\n{result}")

# Test task 3 (contravariant_to_covariant)
print("\nTask 3 - Contravariant to covariant:")
for i, input_data in enumerate(inputs['task1']):  # Uses same inputs as task 1
    result = contravariant_to_covariant(**input_data)
    print(f"Test {i+1}: Shape {result.shape}")
    print(f"Result:\n{result}") 