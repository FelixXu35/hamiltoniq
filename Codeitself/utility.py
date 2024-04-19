"""
Supporting functions
"""

from typing import List, Any, Callable, Dict, Tuple
import random

import numpy as np
from qiskit.quantum_info import Statevector, SparsePauliOp

matrix = Any
circuit = Any
counts = Dict


def Q_to_paulis(Q):
    n_qubits = np.shape(Q)[0]
    offset = np.triu(Q, 0).sum() / 2
    pauli_terms = []
    coeffs = []

    coeffs = -np.sum(Q, axis=1) / 2

    for i in range(n_qubits):
        pauli = ["I" for i in range(n_qubits)]
        pauli[i] = "Z"
        pauli_terms.append("".join(pauli))

    for i in range(n_qubits - 1):
        for j in range(i + 1, n_qubits):
            pauli = ["I" for i in range(n_qubits)]
            pauli[i] = "Z"
            pauli[j] = "Z"
            pauli_terms.append("".join(pauli))

            coeff = Q[i][j] / 2
            coeffs = np.concatenate((coeffs, coeff), axis=None)

    return SparsePauliOp(pauli_terms, coeffs=coeffs), offset


def all_quantum_states(n_qubits):
    states = []
    for i in range(2**n_qubits):
        a = f"{bin(i)[2:]:0>{n_qubits}}"
        vector = [0 for i in range(n_qubits)]
        for i, j in enumerate(a):
            if j == "1":
                vector[i] = 1
        states.append(vector)
    return states


def simple_coupling_map() -> list[Tuple[int, int]]:
    """Return a simple coupling map with 7 qubits.
    This coupling map is used by `ibm_lagos` and `ibm_perth`.
    """

    coupling_map = [[0, 1], [1, 2], [1, 3], [3, 5], [4, 5], [5, 6]]
    return coupling_map
