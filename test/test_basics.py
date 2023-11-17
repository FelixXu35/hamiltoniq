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

def test_all_quantum_states()