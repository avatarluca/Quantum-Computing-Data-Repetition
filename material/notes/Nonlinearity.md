# Non-linearity

In classical NN nonlinearity is introduced by activation functions.
How is it done in the quantum case? Isn't entanglement a nonlinear operation?

Entanglement is not a nonlinear operation in the mathematical sense. In fact quantum circuits themselves are linear operations (unitary operations). 
However, nonlinearity can arise when input data is encoded multiple times.
Entanglement arises when 2 or more quantum systems interact in such a way that their joint state cannot be written as a (tensor) product of individual states.

## Example of entanglement

> |psi1> = a|0> + b|1>, |psi2> = c|0> + d|1>

Joint state (tensor product) withouth entanglement: 

> |psi1>|psi2> = (a|0> + b|1>)(c|0> + d|1>) = ac|00> + ad|01> + bc|10> + bd|11>
(Its still separable)

Now with entanglement (with unitary operation like (XOR = Tensorproduct) CNOT: |a> XOR |b> -> |a> XOR |a XOR b> => flips second qubit if first qubit is 1): 
> |psi_joint> = 1/sqrt(2)(|0> + |1>)|0> = 1/sqrt(2)(|00> + |10>)

> CNOT|psi_joint> = 1/sqrt(2)(|00> + |11>)

(Those 2 qubits are now no longer separable, because the joint state cannot be written as a (tensor) product of individual states)

## What is the nonlinearity in quantum NN?

Nonlinearity in quantum NN is introduced by density matrices (and repetitions of quantum circuits).
Density matrices are used to represent mixed states (superposition of states).

Notes from [Parth G](https://www.youtube.com/watch?v=ZAOc4eMTQiw&ab_channel=ParthG):

Pure states:

> |psi> = |↑>

> |psi> = |↓>

> |psi> = 1/sqrt(2)(|↑> + |↓>) (superposition for example after Hadamard)

Mixed states:
Imagine we don't have enough information to know the state of the qubit. 
We just know that the qubit can be in one of many different pure states with some probability. To describe this we use mixed states.
So there are next to the proability of the basis states |↑> and |↓> in a pure state also probabilities of a pure state on its own.
Imagine that there is a electron source that gives electrons in different psi states.
We don't know the state of the electron, but we know that it can be in one of the pure states with some probability.
We can measure and find the probability of the electron being in a certain psi state (for example every pure state above with 33%).
But we don't know which electron is in which psi state. 
To describe this mathematically we use density matrices.

With density matrice we not only can describe mixed states, but also pure states.

For pure states:

> p = |psi><psi|

For mixed states:

> p = sum_i p_i |psi_i><psi_i|

Where p_i is the probability of the state |psi_i> being in the mixed state.

Note that single vectors / wave functions |psi> can't describe mixed states, that's why we use density matrices.


Example of finding the density matrix p of a pure state:

> |psi> = |↑>

> p = |psi><psi| = (1 0)^T(1 0) = (1 0; 0 0)

Example of finding the density matrix p of a mixed state (it can be in multiple of psi states with some probability):
Idea: Adding up the density matrices of each possible pure state with its probability.


> |psi_1> = |↑>

> |psi_2> = |↓>

> |psi_3> = 1/sqrt(2)(|↑> + |↓>)

> p_final = 1/3 p_1 + 1/3 p_2 + 1/3 p_3 = 1/3(1 0; 0 0) + 1/3(0 0; 0 1) + 1/3(1/2 1/2; 1/2 1/2) = ...

1/3 etc. is the probability of the state being in the pure state. Those are independent.

Now how to introduce nonlinearity in quantum NN with density matrices?
Key: Every matrix with vector operation is linear (like with pure states and for example rotation operation). For nonlinearity we need matrix with matrix operation (for example density matrix and rotation operation)!
Repetition with density matrices (repeating the quantum circuit) introduces nonlinearity in quantum NN, because density matrixes are multiplied with themselves (repeated) and this is a nonlinear operation as described above. When multiplying density matrices p tensorproduct p (by encoding repetition) we get on the diagonal non linear terms.
Attention: In the quantum machine learning part we are now bringing nonlinearity already in the encoding part and not as classical NN in the activation function part after weights and biases.

We can also just repeat only part of the encoding (input features). 
The idea is that they should be uncorrelated (independent) and then we can repeat them. The idea is that features should be uncorralated and later ther combinationes via entanglement create the correlation.

Next to nonlinearity by repeating the encoding (maybe the most easy way) we can also introduce nonlinearity by using ancilla qubits (extra qubits = "ancillas") with tracing out:
In general the density matrix of the ancilla qubit which gets introduced to the system behaves like a normal linear operator (like the rotation operation). The key relies in the tracing out of the ancilla qubit.