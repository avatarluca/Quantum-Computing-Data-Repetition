import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def variational_circuit(data, repetitions):
    """
    Erzeugt einen Quantum Circuit mit datenabhängigen Rotationen und Repetitionen.

    data: Eingabedatenpunkt x (ein float)
    repetitions: Anzahl der Wiederholungen der Daten-Einbettung

    return: QuantumCircuit
    """

    num_qubits = 1
    qc = QuantumCircuit(num_qubits)

    for _ in range(repetitions):
        qc.rx(data, 0)  # Rotation um die x-Achse, skaliert mit dem Datenwert
        qc.ry(data, 0)  # Rotation um die y-Achse

    qc.measure_all()
    return qc


def simulate(data, max_repetitions=5):
    """
    Simuliert den Quantum Circuit für verschiedene Anzahlen an Wiederholungen.
    data: Eingabewert x
    max_repetitions: Maximale Anzahl der Wiederholungen
    """

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
    """
    Plottet die Simulationsergebnisse der Quantum Circuits.
    results: Dictionary mit Counts für verschiedene Repetitionen
    """

    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    fig.suptitle("Impact of Data Repetitions on Quantum Circuit Outputs")

    for i, (reps, counts) in enumerate(results.items()):
        ax = axes[i]
        plot_histogram(counts, ax=ax, title=f"Repetitions: {reps}")

    plt.show()


if __name__ == "__main__":
    # Beispieldaten
    data_input = 1.0  
    max_reps = 10     

    print("Simuliere den Quantum Ansatz mit unterschiedlichen Daten-Repetitionen...")
    results = simulate(data_input, max_repetitions=max_reps)

    plot_results(results)