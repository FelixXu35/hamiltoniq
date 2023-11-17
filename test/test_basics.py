import sys
import os

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)

import pytest
import numpy as np

from HamilToniQ.benchmarking import Toniq
from HamilToniQ.utility import all_quantum_states, Q_to_paulis

tonic = Toniq()

cases_1 = [
    (2, 3, 4),
    (3, 5, 5),
]  # (dim, lower, upper)


@pytest.mark.parametrize("dim, lower, upper", cases_1)
def test_get_Q_matirx(dim, lower, upper):
    Q_matrix = tonic.get_Q_matirx(dim, lower, upper)
    assert isinstance(Q_matrix, np.ndarray)
    assert np.shape(Q_matrix) == (dim, dim)
    assert np.max(Q_matrix) <= upper
    assert np.max(Q_matrix) >= lower
    assert np.array_equal(Q_matrix, Q_matrix.T)


@pytest.mark.parametrize("n_qubits", [3, 4])
def test_all_quantum_states(n_qubits):
    states = all_quantum_states(n_qubits=n_qubits)
    assert np.shape(states)[0] == 2**n_qubits


def test_Q_to_paulis_simple_case():
    Q = np.array([[1, 2], [2, 3]])
    expected_paulis = ["ZI", "IZ", "ZZ"]
    expected_coeffs = [
        -1.5,
        -2.5,
        1.0,
    ]  # Modify as per the correct logic of your function
    expected_offset = 3.0  # Modify as per the correct logic of your function

    # Call the function
    result_pauli_op, result_offset = Q_to_paulis(Q)

    # Assert the results
    assert result_offset == expected_offset
    assert all(
        [a == b for a, b in zip(result_pauli_op.paulis.to_labels(), expected_paulis)]
    )
    assert np.allclose(result_pauli_op.coeffs, expected_coeffs)
