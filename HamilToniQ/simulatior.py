"""
Customized noisy simulators.
"""

from typing import List, Tuple

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
            raise ValueError("T1 doesn`t match with qubits.")
        if np.shape(T2_list)[0] != self.n_qubits:
            raise ValueError("T2 doesn`t match with qubits.")

    def add_readout_error(self, rate_list: list[Tuple[float, float]]) -> None:
        """Add readout error to each qubits.

        args:
            rate_list: a list readout error rates.
                The length of this list should be equal to the number of qubits.
                Each element should be a tuple of two float, corresponding to (p0given1, p1given0)
        """
        # lenght check
        if np.shape(rate_list)[0] != self.n_qubits:
            raise ValueError("Readout error doesn`t match with qubits.")

        # add readout error
        for qubit, [p0given1, p1given0] in enumerate(rate_list):
            readout_error = ReadoutError(
                [[1 - p1given0, p1given0], [p0given1, 1 - p0given1]]
            )
            self.noise_model.add_readout_error(readout_error, qubits=qubit)
            self.noise_model.a
    
    def add_pauli_x_error(self, rate_list: list[float]) -> None:
        pass

    def add_id_error(self, rate_list: list[float]) -> None:
        pass

