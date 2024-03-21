import sys
import os

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)


import pytest
import numpy as np

from else.benchmarking import Toniq

tonic = Toniq()

def test_QAOA_cost()