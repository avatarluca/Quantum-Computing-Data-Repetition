# Input Redundancy for Parameterized Quantum Circuits

The idea is to prepare several quantum registers in a state which is the tensor product of the number of identical copies of the data encoding state. 

Next to binary or amplitude encoding there are also many other ways including tensorial encoding.
Tensorial encoding is the most straightforward way to introduce redundancy.

Goal: Defining lower bounds for the number of redundant copies of the data encoding state for the function the PQC is supposed to learn.

## General idea about non linearity (in addition/recap to Nonlinearity.md)

There exists mainly 2 ways of encoding strategies (parallel and sequential): 
- In Quantum Ansatz where we reupload the data multiple times 
- In Data encoding where we repeat the encoding multiple times sequentially

In data encoding repetition (repeat data encoding several times) we introduce nonlinearity mainly mathematically because of the density matrices, which we can build for pure and mixed states. With can represent pure state with vectors. The "problem" is that vector times matrix (for example rotation matrix like pauli gate used for encoding) is linear. For nonlinearity we need matrix times matrix operation. This is where density matrices come into play which then can introduce nonlinearity.