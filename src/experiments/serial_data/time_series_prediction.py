"""
Dataset Infos:
- Includes multiple companies stock data including Open, Clode, High, Low, Volume
- Dates as timestamps
=> Requires nonlinearity to capture the stock price movements
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from qiskit import ClassicalRegister, QuantumRegister
from qiskit.providers.basic_provider import BasicSimulator  
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA # From Qiskit tutorial
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector


stock_data = pd.read_csv('C:/Users/lucam/OneDrive/Desktop/ZHAW/Semester_5/QI/Project/git/Quantum-Computing-Data-Repetition/src/experiments/serial_data/data/indexData.csv')  # Replace with the actual file path
close_prices = stock_data['Close'].values  # feature = close prices

x = np.arange(len(close_prices)) 



# Still need to implement it. Following code is just experimenting around.
# Create custom feature map or using ZZFeatureMap



# data preprocessing
x_normalized = (x - np.min(x)) / (np.max(x) - np.min(x))  
target_y = close_prices 


def target_function(x, scaling=1, coeffs=None, coeff0=None):
    res = coeff0
    for idx, coeff in enumerate(coeffs):
        exponent = 1j * scaling * (idx + 1) * x
        conj_coeff = np.conjugate(coeff)
        res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
    return np.real(res)


def quantum_model(r, scaling=1):
    """
    Create a quantum circuit for nonlinear regression (Fourier series approximation).
    r: Number of encoding repetitions.
    """

    params = ParameterVector('θ', length=3 * (r + 1))  
    x_param = Parameter('x')  # data input parameter

    qr = QuantumRegister(1)  
    cr = ClassicalRegister(1)  
    qc = QuantumCircuit(qr, cr)

    # draw the circuit
    qc.draw(output='mpl')

    for i in range(r):
        qc.rx(scaling * x_param, qr[0])  # data encoding via R_x
        qc.rz(params[3 * i], qr[0])
        qc.ry(params[3 * i + 1], qr[0])
        qc.rz(params[3 * i + 2], qr[0])

    qc.rz(params[3 * r], qr[0])
    qc.ry(params[3 * r + 1], qr[0])
    qc.rz(params[3 * r + 2], qr[0])

    qc.measure(qr, cr)

    return qc, params, x_param

# From tutorial
def cost_function(params_values, circuit, params, x_param, x_vals, target_vals):
    mse = 0
    sampler = Sampler()
    for x_val, y_target in zip(x_vals, target_vals):
        bound_circuit = circuit.assign_parameters({params: params_values, x_param: x_val})

        result = sampler.run(bound_circuit).result()
        quasi_dists = result.quasi_dists
        
        if len(quasi_dists) == 0 or any(np.isnan(prob) or prob == 0 for _, prob in quasi_dists[0].items()):
            print(f"Error: Invalid quasi distribution for x = {x_val}, skipping")
            mse += 1000000  # Adding a large cost for invalid predictions
            continue
        
        outcome_distribution = list(quasi_dists[0].items())
        
        if len(outcome_distribution) == 0:
            print(f"Error: Empty outcome distribution for x = {x_val}, skipping")
            mse += 1000000  # Adding a large cost for empty distribution
            continue
        
        prediction = int(outcome_distribution[np.argmax([prob for _, prob in outcome_distribution])][0])

        mse += (prediction - y_target) ** 2

    mse /= len(x_vals)
    print(f"Current MSE: {mse}")
    return mse

r = 1  
qc, params, x_param = quantum_model(r)

np.random.seed(42)
initial_params = 2 * np.pi * np.random.rand(len(params))

optimizer = COBYLA(maxiter=20)
result = optimizer.minimize(
    fun=lambda params_vals: cost_function(params_vals, qc, params, x_param, x_normalized, target_y),
    x0=initial_params
)

# print("Optimized parameters:", result.x)

# evaluate the model
predictions = []
backend = BasicSimulator()
sampler = Sampler()

for x_val in x_normalized:
    bound_circuit = qc.assign_parameters({params: result.x, x_param: x_val})
    
    result = sampler.run(bound_circuit).result()
    quasi_dists = result.quasi_dists
    
    outcome_distribution = list(quasi_dists[0].items())
    prediction = int(outcome_distribution[np.argmax([prob for _, prob in outcome_distribution])][0])
    predictions.append(prediction)


plt.plot(x, target_y, c='black', label="Target")
plt.plot(x, predictions, c='blue', label="Prediction")
plt.scatter(x, target_y, facecolor='white', edgecolor='black')
plt.ylim(min(target_y) - 1, max(target_y) + 1)
plt.title("Trained Quantum Model vs Target")
plt.legend()
plt.show()