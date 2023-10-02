"""
This Class is defined to give an overall score of the QAOA performance on a quantum hardware
"""

from typing import Callable, List, Any

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import curve_fit
from scipy import interpolate
from pathlib import Path
from scipy.optimize import minimize
from qiskit import QuantumCircuit, Aer
from qiskit.primitives import Sampler, Estimator, BackendSampler
from qiskit.algorithms.minimum_eigensolvers import QAOA, MinimumEigensolverResult
from qiskit.algorithms.optimizers import COBYLA
from functools import partial

from utility import Q_to_paulis, all_quantum_states
from matrices import *

Matrix = Any
Counts = Any
Circuit = Any
Hardware_Backend = Any


class Toniq:
    def __init__(self) -> None:
        self.backend_list = []
        self.n_reps = 1000
        pass

    def get_Q_matirx(self, dim: int, lower: float = 0.0, upper: float = 10.0):
        """
        Generate a random symmetric matrix with a give dimension.
        args:
            dim: the dimension of Q matrix
            lower: the lower boundary of each random element
            upper: the upper boundary of each random element
            print_hardness: print out the hardness of generated Q matrix
        return:
            mat: a symmetric matrix
        """
        mat = np.array([random.uniform(lower, upper) for _ in range(dim**2)])
        mat = mat.reshape(dim, dim)
        mat = np.triu(mat)
        mat += mat.T - np.diag(mat.diagonal())

        # print the hardness
        normalized_covariance = [
            mat[i, j] / np.sqrt(mat[i, i] * mat[j, j])
            for i in range(len(mat))
            for j in range(i + 1)
        ]
        print(f"the hardness is {np.var(normalized_covariance)}")

        return mat

    def get_ground_state(self, Q) -> dict:
        """
        Find the ground state information of a Q matrix
        args:
            Q: the Q matrix
        return:
            ground: a dict including the ground state in binary form (string), decimal form (int)
            and the corresponding energy (float).
        """
        n_qubits = np.shape(Q)[0]
        energy_list = []
        for state in all_quantum_states(n_qubits):
            energy_list.append(np.dot(state, np.dot(Q, state)))
        dec_min = np.argmin(energy_list)  # ground state in binary form
        ground = {
            "bin_state": f"{bin(dec_min)[2:]:0>{n_qubits}}",
            "dec_state": dec_min,
            "energy": energy_list[dec_min],
        }
        return ground

    def get_results(
        self, backend, Q: Matrix, n_layers: int, options=None, n_reps: int = 1
    ):
        """
        Run QAOA on a given backend.
        args:
            backend: Qiskit backend or simulator
            Q: Q matrix
            n_layers: the number of layers
            options:
            n_reps: for how many times does QAOA run
        return:
            a list of MinimumEigensolverResult
        """
        sampler = BackendSampler(backend=backend, options=options)
        optimizer = COBYLA(maxiter=self.maxiter)
        self.param_list = []
        self.energy_list = []
        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=self.n_layers,
        )
        op, _ = Q_to_paulis(self.Q)
        return [qaoa.compute_minimum_eigenvalue(op) for _ in range(n_reps)]

    def get_reference(
        self, Q: Matrix, n_layers: int, n_reps: int = 10000, n_points: int = 1000
    ) -> list:
        """
        Calculate the scoring function.
        The scoring function is represented by uniform sampling.

        args:
            Q:
            n_layers:
            n_reps: number of repetation by which the QAOA run on a simulator
            n_points: number of points in sampling percedure
        return:
            scoring_curve_sampling: A list of uniform sampling of the scoring curve. It has 201 elements.
            The corresponding x-axis is build using `np.linspace(0, 1, 201)`.
        """
        ground_state_info = self.get_ground_state(Q)
        dec_ground_state = ground_state_info["dec_state"]
        backend = Aer.get_backend("aer_simulator")

        # get the distribution of accuracy
        results = self.get_results(backend, Q, n_layers, n_reps=n_points)
        accuracy_list = []
        for i in results:
            try:
                accuracy_list.append(i.eigenstate[dec_ground_state])
            except:
                accuracy_list.append(0)
        n_boxes = 200
        hist_x = np.linspace(0, 1, n_boxes + 1)
        hist_y, _ = np.histogram(accuracy_list, bins=hist_x)
        hist_y = np.divide(hist_y, accuracy_list.size)

        # build the scoring function
        cumulative_score = np.cumsum(hist_y)
        scoring_curve_sampling = np.append(np.zeros(1), cumulative_score)
        return scoring_curve_sampling

    def score(accuracy_data: List(float), dim, n_layers) -> float:
        df = pd.read_csv("HamilToniQ/scoring_curves.csv")
        score_y = df[f"dim_{dim}_layer_{n_layers}"]
        score_x = df["score_x"]
        f = interpolate.interp1d(score_x, score_y, kind="linear")
        score = 0.0
        n_points = np.shape(accuracy_data)[0]
        for i in accuracy_data:
            score += f(i) / n_points
        return score

    def get_accuracy(
        self, data: List(MinimumEigensolverResult), dim: int, n_layers: int
    ) -> List(float):
        """
        Calculate the accuracy (overlap between the result and the ground state) for all QAOA results.
        """
        ground_state_info = globals(f"ground_{dim}")
        dec_ground_state = ground_state_info["dec_state"]
        accuracy_list = []
        for i in data:
            try:
                accuracy_list.append(i.eigenstate[dec_ground_state])
            except:
                accuracy_list.append(0)
        return accuracy_list

    def run(self, backend, dim: int, n_layers: int) -> float:
        """
        Score backend with a specific width/dimension and a number of layers.
        args:
            backend: the qiskit backend that is going to be scored.
        """
        results_list = self.get_results(
            backend, globals(f"dim_{dim}"), n_layers, n_reps=1000
        )
        accuracy_list = self.get_accuracy(results_list, dim, n_layers)
        return self.score(accuracy_list)

    def show_ladder_diagram(self, dim: int, n_layers: int, backends=None):
        f"""
        Show the benchmarking results obtained by us.
        {self.backend_list}
        """
        if backends in self.backend_list:
            pass
        pass
