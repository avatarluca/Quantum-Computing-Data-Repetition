"""
Dataset Infos:
- MNIST dataset with 28x28 pixel images of handwritten digits
=> We gonna use QSVM to classify the digits

Main idea:
Trying to classify handwritten digits from the MNIST dataset
Goal is to differentiate between '0' to '9' 
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC

def load_mnist_data():
    from sklearn.datasets import load_digits
    digits = load_digits() 
    return digits.data, digits.target

X, y = load_mnist_data()
scaler = PCA(n_components=10)  
X_reduced = scaler.fit_transform(X)

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler(feature_range=(0, 2 * np.pi))
X_scaled = mms.fit_transform(X_reduced)

repetitions = [1, 3, 5]
encodings = {
    "Linear Encoding": PauliFeatureMap(feature_dimension=X_scaled.shape[1], paulis=['Z']),
    "Arcsine Encoding": ZZFeatureMap(feature_dimension=X_scaled.shape[1], entanglement='circular'),
}

def calculate_fourier_rank(kernel_matrix):
    return np.linalg.matrix_rank(kernel_matrix)

for encoding_name, feature_map in encodings.items():
    print(f"\nEncoding: {encoding_name}")
    for r in repetitions:
        print(f"  Training with repetition rate: {r}")
        
        feature_map.reps = r
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
        
        kernel_matrix = quantum_kernel.evaluate(x_vec=X_scaled)

        fourier_rank = calculate_fourier_rank(kernel_matrix)
        print(f"    Fourier rank: {fourier_rank}")
        
        svm = SVC(kernel='precomputed', decision_function_shape='ovr')
        svm.fit(kernel_matrix, y)
        
        y_pred = svm.predict(kernel_matrix)
        accuracy = accuracy_score(y, y_pred)
        print(f"    Accuracy: {accuracy:.4f}")
        
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, cmap='Blues')
        plt.title(f"Confusion Matrix (Reps={r}, {encoding_name})")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()
        
        reduced_kernel = PCA(n_components=2).fit_transform(kernel_matrix)
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_kernel[:, 0], reduced_kernel[:, 1], c=y, cmap='viridis', s=50)
        plt.title(f"Feature Space (Reps={r}, {encoding_name})")
        plt.colorbar()
        plt.show()
        
        plt.figure(figsize=(8, 6))
        plt.imshow(kernel_matrix, cmap='hot', interpolation='nearest')
        plt.title(f"Kernel Matrix (Reps={r}, {encoding_name})")
        plt.colorbar()
        plt.show()
