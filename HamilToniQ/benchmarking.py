"""
This Class is defined to give an overall score of the QAOA performance on a quantum hardware
"""

from typing import Callable, List, Any, Sequence

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import cpu_count, Pool
from itertools import product
from scipy.optimize import curve_fit
from scipy import interpolate
from pathlib import Path
from scipy.optimize import minimize, OptimizeResult
from qiskit import Aer
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BackendSampler
from qiskit.circuit.library import QAOAAnsatz
from qiskit_algorithms.minimum_eigensolvers import QAOA, MinimumEigensolverResult
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import MinimumEigensolverResult
from qiskit_ibm_runtime import Options, Session, Estimator
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_ibm_runtime import Estimator
from functools import partial

from .utility import Q_to_paulis, all_quantum_states
from .matrices import *

Matrix = Any
Counts = Any
Hardware_Backend = Any


class Toniq:
    def __init__(self) -> None:
        self.backend_list = []
        self.maxiter = 10000

    def get_Q_matirx(
        self, dim: int, lower: float = -10.0, upper: float = 10.0
    ) -> Matrix:
        """Generate a random symmetric matrix with a give dimension.

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

    def QAOA_cost(
        self,
        params: Sequence[float],
        ansatz: QuantumCircuit,
        op: SparsePauliOp,
        estimator: Estimator,
    ):
        """Return estimate of energy from estimator

        Parameters:
            params (ndarray): Array of ansatz parameters
            ansatz (QuantumCircuit): Parameterized ansatz circuit
            hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
            estimator (Estimator): Estimator primitive instance

        Returns:
            float: Energy estimate
        """
        cost = estimator.run(ansatz, op, parameter_values=params).result().values[0]
        return cost

    def get_ground_state(self, Q) -> dict:
        """Find the ground state information of a Q matrix

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

    def get_results_simulator(
        self,
        fake_backend,
        Q: Matrix,
        n_layers: int,
        options=None,
        n_reps: int = 1000,
        n_cores: int | None = None,
    ) -> Sequence[MinimumEigensolverResult]:
        """Run QAOA on a given fake backend.

        args:
            fake_backend: Qiskit fake backend, which is a noisy simulator
            Q: Q matrix
            n_layers: the number of layers
            options:
            n_reps: for how many times does QAOA run

        return:
            a list of MinimumEigensolverResult
        """
        if n_cores is None:
            n_cores = cpu_count()
        sampler = BackendSampler(backend=fake_backend, options=options)
        optimizer = COBYLA(maxiter=self.maxiter)
        self.param_list = []
        self.energy_list = []
        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=n_layers,
        )
        op, _ = Q_to_paulis(Q)
        with Pool(8) as p:
            results = p.map(
                qaoa.compute_minimum_eigenvalue, [op for _ in range(n_reps)]
            )
        return results

    def get_results_processor(
        self,
        backend,
        Q: Matrix,
        n_layers: int,
        options=None,
        n_reps: int = 1000,
        resiliance=0,
    ) -> Sequence[OptimizeResult]:
        op, _ = Q_to_paulis(Q)
        ansatz = QAOAAnsatz(op, reps=n_layers)
        session = Session(backend=backend)
        options = Options()
        options.resilience_level = resiliance
        estimator = Estimator(session=session, options=options)
        x0 = (
            np.pi * np.random.rand(ansatz.num_parameters) - np.pi / 2
        )  # the same bounds as `SamplingVQE` class
        results = [
            minimize(self.QAOA_cost, x0, args=(ansatz, op, estimator), method="COBYLA")
            for _ in range(n_reps)
        ]
        return results

    def get_reference(
        self,
        Q: Matrix,
        n_layers: int,
        n_points: int = 10000,
        n_cores=1,
    ) -> Sequence[float]:
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
        results = self.get_results_simulator(
            backend, Q, n_layers, n_reps=n_points, n_cores=n_cores
        )
        accuracy_list = []
        for i in results:
            try:
                accuracy_list.append(i.eigenstate[dec_ground_state])
            except:
                accuracy_list.append(0)
        n_boxes = 200
        hist_x = np.linspace(0, 1, n_boxes + 1)
        hist_y, _ = np.histogram(accuracy_list, bins=hist_x)
        hist_y = np.divide(hist_y, np.shape(accuracy_list)[0])

        # build the scoring function
        cumulative_score = np.cumsum(hist_y)
        scoring_curve_sampling = np.append(np.zeros(1), cumulative_score)
        return scoring_curve_sampling

    def score(self, accuracy_data: list, dim, n_layers) -> float:
        df = pd.read_csv("HamilToniQ/scoring_curves.csv")
        score_y = df[f"dim_{dim}_layer_{n_layers}"]
        score_x = df["score_x"]
        f = interpolate.interp1d(score_x, score_y, kind="linear")
        score = 0.0
        n_points = np.shape(accuracy_data)[0]
        for i in accuracy_data:
            score += f(i) * 2 / n_points
        return score

    def get_accuracy_simulator(
        self, data: Sequence[MinimumEigensolverResult], dim: int
    ) -> Sequence[float]:
        """
        Calculate the accuracy (overlap between the result and the ground state) for all QAOA results.
        """
        ground_state_info = globals()[f"ground_{dim}"]
        dec_ground_state = ground_state_info["dec_state"]
        accuracy_list = []
        for res in data:
            try:
                accuracy_list.append(res.eigenstate[dec_ground_state])
            except:
                accuracy_list.append(0)
        return accuracy_list

    def get_accuracy_processor(
        self, data: Sequence[OptimizeResult], dim: int, n_layers: int, Q
    ) -> Sequence[float]:
        op, _ = Q_to_paulis(Q)
        ansatz = QAOAAnsatz(op, reps=n_layers)
        ground_state_info = globals()[f"ground_{dim}"]
        dec_ground_state = ground_state_info["dec_state"]
        accuracy_list = []
        for res in data:
            qc = ansatz.bind_parameters(res.x)
            sv = Statevector(qc)
            accuracy_list.append(abs(sv[dec_ground_state]) ** 2)
        return accuracy_list

    def simulator_run(
        self,
        fake_backend,
        dim: int,
        n_layers: int,
        n_cores: int | None = None,
        n_reps: int = 1000,
    ) -> float:
        """
        Score a backend with a specific width/dimension and a number of layers.
        args:
            backend: the qiskit backend that is going to be scored.
        """
        results_list = self.get_results_simulator(
            fake_backend,
            globals()[f"dim_{dim}"],
            n_layers,
            n_reps=n_reps,
            n_cores=n_cores,
        )
        accuracy_list = self.get_accuracy_simulator(results_list, dim)
        return self.score(accuracy_list, dim=dim, n_layers=n_layers)

    def processor_run(
        self, backend, dim: int, n_layers: int, n_reps: int = 1000
    ) -> float:
        results_list = self.get_results_processor(
            backend,
            globals()[f"dim_{dim}"],
            n_layers,
            n_reps=n_reps,
        )
        accuracy_list = self.get_accuracy_processor(
            results_list, dim, n_layers, globals()[f"dim_{dim}"]
        )
        return self.score(accuracy_list, dim=dim, n_layers=n_layers)

    def plot_heatmap(self, data: pd.DataFrame, sort_dim: int = 1):
        """
        Use a heatmap to compare across different backend with the same dimension.
        """
        if sort_dim not in range(1, 10):
            raise ValueError("Invalid dimension")
        first_score = data.index[sort_dim]
        data = data.transpose()
        data_sorted = data.sort_values(by=first_score, ascending=False)
        sns.heatmap(
            data_sorted,
            annot=True,
            square=True,
            cbar=False,
            vmax=1,
            vmin=0,
        )

    def show_ladder_diagram(self, dim: int, n_layers: int, backends=None):
        f"""
        Show the benchmarking results obtained by us.
        {self.backend_list}
        """
        if backends in self.backend_list:
            pass
        pass

    def ZNE_settings(self) -> None:
        print("ZNE settings")
