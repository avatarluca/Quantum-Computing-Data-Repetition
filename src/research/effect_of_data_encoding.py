import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)  
theta = np.random.rand(3)  

# Pauli-Z matrix
H = np.array([[1, 0], [0, -1]])

eigenvalues, _ = np.linalg.eigh(H) # 1 and -1 because 2x2 matrix
frequencies = np.subtract.outer(eigenvalues, eigenvalues).flatten()

def trainable_coefficients(theta, num_frequencies):
    return np.array([np.cos(t) + 1j * np.sin(t) for t in theta[:num_frequencies]])

def quantum_model(x, theta, frequencies):
    coeffs = trainable_coefficients(theta, len(frequencies)) 
    return np.real(np.sum(coeffs[:, None] * np.exp(1j * frequencies[:, None] * x), axis=0))

frequencies = frequencies[:3] 

output = quantum_model(x, theta, frequencies)

plt.plot(x, output, label="Quantum Model Output")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()