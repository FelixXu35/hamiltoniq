"""
This Class is defined to give an overall score of the QAOA performance on a quantum hardware
"""

from typing import Callable, List, Any, Sequence

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from multiprocessing import cpu_count, Pool
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.optimize import minimize, OptimizeResult
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
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
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from .utility import (
    Q_to_paulis,
    all_quantum_states,
    get_transpiled_index_layout,
    reorder_bits,
)
from .instances import *

Matrix = Any
Counts = Any
Hardware_Backend = Any
Fake_Backend = Any
Quantum_Processor = Any


class Toniq:
    def __init__(self) -> None:
        self.backend_list = []
        self.maxiter = 10000
        self.options = {"optimization_level": 0, "resilience_level": 0}

    @staticmethod
    def get_Q_matirx(
        n_qubits: int, lower: float | int = -10.0, upper: float | int = 10.0
    ) -> np.ndarray:
        """Generate a random symmetric matrix with a give dimension.

        args:
            n_qubits: the dimension of Q matrix
            lower: the lower boundary of each random element
            upper: the upper boundary of each random element

        return:
            mat: a symmetric matrix
        """
        mat = np.array(
            [random.uniform(lower, upper) for _ in range(n_qubits**2)]
        )  # Create a flat array of random numbers with size n_qubits^2
        mat = mat.reshape(n_qubits, n_qubits)
        mat = np.triu(mat)
        mat += mat.T - np.diag(
            mat.diagonal()
        )  # Make the matrix symmetric by adding its transpose and subtracting the diagonal

        # print the hardness
        normalized_covariance = [
            mat[i, j] / np.sqrt(abs(mat[i, i] * mat[j, j]))
            for i in range(len(mat))
            for j in range(i + 1)
        ]
        print(f"the hardness is {np.var(normalized_covariance)}")

        return mat

    @staticmethod
    def QAOA_cost(
        params: Sequence[float],
        ansatz: QuantumCircuit,
        op: SparsePauliOp,
        estimator: Estimator,
    ) -> float:
        """Return the estimated energy from estimator

        Parameters:
            params: values of ansatz parameters
            ansatz: the quantum circuit from which a state is generated
            hamiltonian: the Hamiltonian to which the energy corresponds
            estimator: estimator primitive instance

        Returns:
            float: estimated energy
        """
        cost = estimator.run(ansatz, op, parameter_values=params).result().values[0]
        return cost

    def set_optimization_level(self, optimization_level: int):
        self.options["optimization_level"] = optimization_level

    def set_resilience_level(self, resilience_level: int):
        self.options["resilience_level"] = resilience_level

    def get_ground_state(self, Q: np.ndarray) -> dict[str, any]:
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
            energy_list.append(
                np.dot(state, np.dot(Q, state))
            )  # calculate all possible energy
        dec_min = np.argmin(energy_list)  # ground state in decimal
        ground = {
            "bin_state": f"{bin(dec_min)[2:]:0>{n_qubits}}",  # ground state in binary
            "dec_state": dec_min,
            "energy": energy_list[dec_min],  # ground state energy
        }
        return ground

    def get_results_simulator(
        self,
        fake_backend: Fake_Backend,
        n_layers: int,
        n_reps: int = 1000,
        n_cores: int | None = None,
    ) -> Sequence[MinimumEigensolverResult]:
        """Return certain number of QAOA results on a specified fake backend.

        args:
            fake_backend: Qiskit fake backend, which is a noisy simulator
            n_layers: the number of layers
            n_reps: how many QAOA results will be returned

        return:
            a list of MinimumEigensolverResult
        """
        if n_cores is None:
            n_cores = cpu_count()  # detect the total number of cores
        sampler = BackendSampler(backend=fake_backend, options=self.options)
        optimizer = COBYLA(maxiter=self.maxiter)
        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=n_layers,
        )
        with Pool(n_cores) as p:
            results = p.map(
                qaoa.compute_minimum_eigenvalue, [self.op for _ in range(n_reps)]
            )
        return results

    def get_results_processor(
        self,
        backend: Quantum_Processor,
        n_layers: int,
        n_reps: int = 1000,
    ) -> Sequence[OptimizeResult]:
        """Return certain number of QAOA results on a specified quantum processor.

        args:
            backend: IBM Quantum Computer
            n_layers: the number of layers
            n_reps: how many QAOA results will be returned

        return:
            a list of MinimumEigensolverResult
        """
        self.ansatz = QAOAAnsatz(self.op, reps=n_layers)

        # Optimize ISA circuit for quantum execution
        target = backend.target
        self.pm = generate_preset_pass_manager(
            target=target, optimization_level=self.options["optimization_level"]
        )
        self.ansatz_isa = self.pm.run(self.ansatz)

        # ISA observable
        self.op_isa = self.op.apply_layout(self.ansatz_isa.layout)

        session = Session(backend=backend)
        estimator = Estimator(session=session)
        x0 = (
            np.pi * np.random.rand(self.ansatz_isa.num_parameters) - np.pi / 2
        )  # the same bounds as `SamplingVQE` class
        results = [
            minimize(
                self.QAOA_cost,
                x0,
                args=(self.ansatz_isa, self.op_isa, estimator),
                method="COBYLA",
            )
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
        """Calculate the score function.
        The score function is represented by uniform sampling.

        args:
            Q:
            n_layers:
            n_reps: number of repetation by which the QAOA run on a simulator
            n_points: number of points in sampling percedure

        return:
            score_curve_sampling: A list of uniform sampling of the score curve. It has 201 elements.
            The corresponding x-axis is build using `np.linspace(0, 1, 201)`.
        """
        # prepare Q-matrix and its operators
        self.Q = Q
        self.op, _ = Q_to_paulis(self.Q)

        ground_state_info = self.get_ground_state(Q)
        dec_ground_state = ground_state_info["dec_state"]
        backend = AerSimulator()

        # get the distribution of accuracy
        results = self.get_results_simulator(
            backend, n_layers, n_reps=n_points, n_cores=n_cores
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

        # build the score function
        cumulative_score = np.cumsum(hist_y)
        score_curve_sampling = np.append(np.zeros(1), cumulative_score)
        return score_curve_sampling

    def score(self, accuracy_data: list, n_qubits: int, n_layers: int) -> float:
        """Calculate the score according to the QAOA accuracy data and the score curves.

        args:
            accuracy_data: the accuracy obtained from QAOA results
            n_qubits: the number of qubits (used to find the score curves)
            n_layers: the number of layers (used to find the score curves)

        return:
            score: the score of a backend
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        csv_file_path = os.path.join(dir_path, "score_curves.csv")
        df = pd.read_csv(csv_file_path)  # import the score curve
        score_y = df[f"qubits_{n_qubits}_layer_{n_layers}"]
        score_x = df["score_x"]
        f = interpolate.interp1d(
            score_x, score_y, kind="linear"
        )  # build the score function
        score = 0.0
        n_data = np.shape(accuracy_data)[0]
        for i in accuracy_data:
            score += f(i) * 2 / n_data
        return score

    def get_accuracy_simulator(
        self, data: Sequence[MinimumEigensolverResult], n_qubits: int
    ) -> Sequence[float]:
        """Analyse the QAOA results from a simulator and extract their accuracy.

        args:
            data: a list of QAOA results obtained from a simualtor
            n_qubits: the number of qubits

        return:
            accuracy_list: a list of accuracy, corresponding to input results
        """
        ground_state_info = globals()[f"ground_{n_qubits}"]
        dec_ground_state = ground_state_info["dec_state"]
        accuracy_list = []
        for res in data:
            try:
                accuracy_list.append(res.eigenstate[dec_ground_state])
            except:
                accuracy_list.append(0)
        return accuracy_list

    def get_accuracy_processor(
        self,
        data: Sequence[OptimizeResult],
        n_qubits: int,
        n_layers: int,
    ) -> Sequence[float]:
        """Analyse the QAOA results from a processor and extract their accuracy.

        args:
            data: a list of QAOA results obtained from a simualtor
            n_qubits: the number of qubits
            n_layers: the number of layers

        return:
            accuracy_list: a list of accuracy, corresponding to input results
        """
        ansatz = QAOAAnsatz(self.op, reps=n_layers)
        ground_state_info = globals()[f"ground_{n_qubits}"]
        bin_ground_state = ground_state_info["bin_state"]
        accuracy_list = []
        for res in data:
            # Replace bind_parameters with assign_parameters
            qc = ansatz.assign_parameters(res.x)
            qc_isa = self.pm.run(qc)
            sv = Statevector(qc_isa)
            # Convert dec_state based on transpiled circuit layout
            new_layout_order = get_transpiled_index_layout(qc_isa)
            _, new_dec_ground_state = reorder_bits(bin_ground_state, new_layout_order)
            accuracy_list.append(abs(sv[new_dec_ground_state]) ** 2)
        return accuracy_list

    def simulator_run(
        self,
        fake_backend,
        n_qubits: int,
        n_layers: int,
        n_cores: int | None = None,
        n_reps: int = 1000,
        Q: np.ndarray | None = None,
        plot_results: bool = False,
    ) -> float:
        """Score a backend with a specific number of qubits and a number of layers.
        This function is dedicated to simulators, since multiprocessing is used to speed up.

        args:
            fake_backend: the qiskit backend that is going to be scored.
            n_qubits: the number of qubits.
            n_layers: the number of layers in QAOA ansatz.
            n_cores: the expected number of cores on PC. Auto detection will be used if this number is not specify.
            n_reps: for how many times the QAOA algorithm is run. The default number is 1000.
            Q: the used-defined Q-matrix used in benchmarking. The default ones will be used if this is not specified.

        return:
            score: the score of this simuator
        """
        # prepare Q-matrix and its operators
        self.Q = globals()[f"qubits_{n_qubits}"]
        if Q is not None:
            self.Q = Q
        self.op, _ = Q_to_paulis(self.Q)

        # run QAOA and get results
        results_list = self.get_results_simulator(
            fake_backend,
            n_layers,
            n_reps=n_reps,
            n_cores=n_cores,
        )

        # analyse the results and get a score
        accuracy_list = self.get_accuracy_simulator(results_list, n_qubits)

        # plot the accuracy list
        if plot_results is True:
            plt.plot(np.sort(accuracy_list))
        return (
            results_list,
            accuracy_list,
            self.score(accuracy_list, n_qubits=n_qubits, n_layers=n_layers),
        )

    def processor_run(
        self,
        backend,
        n_qubits: int,
        n_layers: int,
        n_reps: int = 1000,
        Q: np.ndarray | None = None,
        plot_results: bool = False,
    ) -> float:
        """Score a backend with a specific number of qubits and a number of layers.
        This function is dedicated to real quantum computors.
        Scoring simulators using this function is feasible, but it will take much longer time.

        args:
            fake_backend: the qiskit backend that is going to be scored.
            n_qubits: the number of qubits.
            n_layers: the number of layers in QAOA ansatz.
            n_cores: the expected number of cores on PC. Auto detection will be used if this number is not specify.
            Q: the used-defined Q-matrix used in benchmarking. The default ones will be used if this is not specified.

        return:
            score: the score of this simuator
        """
        # prepare Q-matrix and its operators
        self.Q = globals()[f"qubits_{n_qubits}"]
        if Q is not None:
            self.Q = Q
        self.op, _ = Q_to_paulis(self.Q)

        # run QAOA and get the resutls
        results_list = self.get_results_processor(
            backend,
            n_layers,
            n_reps=n_reps,
        )

        # analyse the results and get a score
        accuracy_list = self.get_accuracy_processor(
            results_list, n_qubits, n_layers, globals()[f"qubits_{n_qubits}"]
        )

        # plot the accuracy list
        if plot_results is True:
            plt.plot(np.sort(accuracy_list))
        return self.score(accuracy_list, n_qubits=n_qubits, n_layers=n_layers)

    def plot_heatmap(self, data: pd.DataFrame, sort_qubit: int = 1) -> None:
        """Use a heatmap to compare across different backend with the same number of qubits.
        This function has no return, but plots a sorted chart.

        args:
            data: scores
            sort_qubit: the qubits with which the chart is sorted.
        """
        if sort_qubit not in range(1, 10):
            raise ValueError("Invalid n_qubitsension")  # shouldn't exceed the range
        first_score = data.index[sort_qubit]  # select the colume
        data = data.transpose()
        data_sorted = data.sort_values(by=first_score, ascending=False)  # sort the data
        sns.heatmap(
            data_sorted,
            annot=True,
            square=True,
            cbar=False,
            vmax=1,
            vmin=0,
        )

    def show_ladder_diagram(self, n_qubits: int, n_layers: int, backends=None):
        f"""
        Show the benchmarking results obtained by us.
        {self.backend_list}
        """
        if backends in self.backend_list:
            pass
        pass
