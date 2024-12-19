import numpy as np
import matplotlib.pyplot as plt



"""
Fourier Rank
-------------

Determines how many basis functiones are needed to represent a function with a tolerance of epsilon.
"""

EPS = 1e-5

def f(x): return np.sin(2 * np.pi * 1 * x) + np.sin(2 * np.pi * 2 * x)

def calculate_fourier_rank(f, x_values):
    y_values = f(x_values)
    fourier_transform = np.fft.fft(y_values) # fast fourier transform
    fourier_rank = np.count_nonzero(np.abs(fourier_transform) > EPS)  
    return fourier_rank

x_values = np.linspace(0, 1, 1000)
fourier_rank = calculate_fourier_rank(f, x_values)
print(f"Fourier Rank: {fourier_rank}")



"""
Linear & arcsine encoding
-------------------------
"""

# examples
a = np.array([1, 2, 3]) # weights
b = np.array([0, 0, 0]) # bias
x = np.array([0.5, 0.2, 0.9])  # input

def linear_encoding(x, a, b): return x * a + b
def arcsine_encoding(x, a, b): return np.arcsin((x * a + b) / (2 * np.pi))

encoded_values = linear_encoding(x, a, b)
arcsine_encoded_values = arcsine_encoding(x, a, b)



"""
Optimal number of repetitions
-----------------------------

Important theorems:
- Fourier Rank: The Fourier Rank of a function gives an upper bound for the input redundancy.
- Input Redundancy Bound: The Input Rank is minimal log3(r + 1) for linear encoding, where r is the fourier rank.
"""

def log3(x):
    return np.log(x) / np.log(3)

def optimal_repetitions(fourier_rank):
    return np.ceil(log3(fourier_rank + 1))

optimal_n = optimal_repetitions(fourier_rank)
print(f"Optimal amount of repetitions: {optimal_n}")


"""
Short overview of different data types
---------------------------------------
"""

# 1. Serial data (like time series stocks data, texts, voice, etc.)
def stock_data_example(t): return np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.sin(2 * np.pi * 0.5 * t)

# timeseries
time_values = np.linspace(0, 10, 100)
stock_prices = stock_data_example(time_values)

fourier_rank_stock = calculate_fourier_rank(stock_data_example, time_values)
optimal_n_stock = optimal_repetitions(fourier_rank_stock)
print(f"Optimal repetiton: {optimal_n_stock}")


# 2. Image data
from skimage import data
from skimage.transform import resize

# example image data
image = data.camera()
image_resized = resize(image, (64, 64))  


def image_fourier_rank(image):
    return np.count_nonzero(np.abs(np.fft.fft2(image)) > 1e-5)

fourier_rank_image = image_fourier_rank(image_resized)
optimal_n_image = optimal_repetitions(fourier_rank_image)
print(f"Optimal repetition: {optimal_n_image}")


# 3. Tabular data (order doesn't matter) TODO
import pandas as pd

# Beispiel-Daten
data = pd.DataFrame({
    'Feature1': np.random.rand(100),
    'Feature2': np.random.rand(100),
    'Feature3': np.random.rand(100)
})

def tabular_fourier_rank(data):
    return [calculate_fourier_rank(lambda x: f, np.arange(len(f))) for f in data.values.T]

fourier_rank_tabular = tabular_fourier_rank(data)

optimal_n_tabular = [optimal_repetitions(r) for r in fourier_rank_tabular]

print(f"Optimal repetition: {optimal_n_tabular}")