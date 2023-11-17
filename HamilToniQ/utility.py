"""
Supporting functions
"""

from typing import List, Any, Callable, Dict
import random

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from qiskit import QuantumCircuit, transpile, Aer, execute
from qiskit.providers.fake_provider import FakeBackendV2
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import IBMBackend
from qiskit.result.counts import Counts
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


def all_quantum_states(n_qubits, budget=None, vec=False):
    states = []
    for i in range(2**n_qubits):
        a = f"{bin(i)[2:]:0>{n_qubits}}"
        n_ones = 0
        vector = [0 for i in range(n_qubits)]
        for i, j in enumerate(a):
            if j == "1":
                vector[i] = 1
        states.append(vector)
    return states
