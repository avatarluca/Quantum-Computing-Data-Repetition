from qiskit import QuantumCircuit, transpile
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def variational_circuit(data, repetitions):
    num_qubits = 1
    qc = QuantumCircuit(num_qubits)

    for _ in range(repetitions):
        qc.rx(data, 0) 
        qc.ry(data, 0) 

    qc.measure_all()
    return qc


def simulate(data, max_repetitions=5):
    backend = BasicSimulator()
    results = {}

    for reps in range(1, max_repetitions + 1):
        qc = variational_circuit(data, repetitions=reps)
        transpiled_qc = transpile(qc, backend)
        job = backend.run(transpiled_qc, shots=8192)
        result = job.result()
        counts = result.get_counts()
        results[reps] = counts

    return results


def plot_results(results):
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    fig.suptitle("Impact of data repetitions on QC outputs")

    for i, (reps, counts) in enumerate(results.items()):
        ax = axes[i]
        plot_histogram(counts, ax=ax, title=f"Repetitions: {reps}")

    plt.show()


if __name__ == "__main__":
    data_input = 1.0  
    max_reps = 10     

    print("Simulate quantum ansatz with different repetitions")
    results = simulate(data_input, max_repetitions=max_reps)

    plot_results(results)