"""
Customized noisy simulators.
"""

from typing import List

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    pauli_error,
    depolarizing_error,
    thermal_relaxation_error,
)


class CustomizeSim:
    def __init__(self, n_qubits: int) -> None:
        self.noise_model = NoiseModel()
        self.n_qubits = n_qubits

    def add_decoherence(self, T1_list: list[float], T2_list: list[float]):
        # lenght check
        if np.shape(T1_list)[0] != self.n_qubits:
            raise ValueError("T1 doesn`t match with qubits")
        if np.shape(T2_list)[0] != self.n_qubits:
            raise ValueError("T2 doesn`t match with qubits")

    def add_readout_error(self):
        pass
