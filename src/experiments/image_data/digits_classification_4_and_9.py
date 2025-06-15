import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC

def load_mnist_data():
    from sklearn.datasets import load_digits
    digits = load_digits(n_class=2) 
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

for encoding_name, feature_map in encodings.items():
    print(f"\nEncoding: {encoding_name}")
    for r in repetitions:
        print(f"  Training with repetition rate: {r}")
        
        feature_map.reps = r
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
        
        kernel_matrix = quantum_kernel.evaluate(x_vec=X_scaled, y_vec=X_scaled)  # kernel between X and X
        
        svm = SVC(kernel='precomputed')
        svm.fit(kernel_matrix, y)
        
        y_pred = svm.predict(kernel_matrix)
        accuracy = accuracy_score(y, y_pred)
        print(f"    Accuracy: {accuracy:.4f}")
        
        # PCA for Visualization
        reduced_kernel = PCA(n_components=2).fit_transform(kernel_matrix)
        grid_x, grid_y = np.meshgrid(np.linspace(reduced_kernel[:, 0].min(), reduced_kernel[:, 0].max(), 100),
                                     np.linspace(reduced_kernel[:, 1].min(), reduced_kernel[:, 1].max(), 100))
        grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
        
        # Compute kernel between grid points and training data
        grid_kernel_matrix = quantum_kernel.evaluate(x_vec=grid_points, y_vec=X_scaled)  # grid_points vs X_scaled
        
        # Compute decision values for grid points
        grid_decision_values = svm.decision_function(grid_kernel_matrix).reshape(grid_x.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(grid_x, grid_y, grid_decision_values, levels=50, cmap='coolwarm', alpha=0.8)
        plt.contour(grid_x, grid_y, grid_decision_values, levels=[0], colors='black', linewidths=2, linestyles='dashed')

        plt.scatter(reduced_kernel[:, 0], reduced_kernel[:, 1], c=y, cmap='viridis', edgecolor='k', s=100, label='Data Points')
        plt.colorbar(label='Decision Value')
        plt.title(f"Decision Boundary (Reps={r}, {encoding_name})")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.imshow(kernel_matrix, cmap='hot')
        plt.title(f"Kernel Matrix (Reps={r}, {encoding_name})")
        plt.colorbar()
        plt.show()
