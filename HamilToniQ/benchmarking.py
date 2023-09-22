"""
This Class is defined to give an overall score of the QAOA performance on a quantum hardware
"""

from typing import Callable, List, Any

import random
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
from scipy.optimize import minimize
from qiskit import QuantumCircuit, Aer
from qiskit.primitives import Sampler, Estimator, BackendSampler
from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit.algorithms.optimizers import COBYLA
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd

from utility import (
    build_cost_func,
    load_qiskit_sampler,
    load_circuit,
    plot_energy_distribution,
    get_overlap,
    Q_to_paulis,
    store_process,
    symmetric_matrix_generator,
    ground_state
)
from matrices import *

Counts = Any
Circuit = Any
Hardware_Backend = Any


class Toniq:
    def __init__(self) -> None:
        self.Q = dim_4_var_6
        self.le = le_4_6
        self.rep = 1000
        self.n_layers = 6
        self.maxiter = 1000
        pass

    def standard_QAOA(self, backend: Hardware_Backend) -> None:
        # sampler = load_qiskit_sampler(backend, 1000)
        cost_func = build_cost_func(backend, self.Q, self.n_layers)
        optimizer = COBYLA(maxiter=self.maxiter)

        def get_result():
            init_params = [
                random.random() * 2 * np.pi for i in range(2 * self.n_layers)
            ]
            result = optimizer.minimize(fun=cost_func, x0=init_params)
            loaded_circuit = load_circuit(self.Q, self.n_layers, result.x)
            return get_overlap(loaded_circuit, self.le)

        # plot_energy_distribution(loaded_circuit, self.Q)
        overlaps = [get_result() for i in range(self.rep)]
        return overlaps

    def default_QAOA(self, backend, options=None):
        sampler = BackendSampler(backend=backend, options=options)
        optimizer = COBYLA(maxiter=self.maxiter)
        self.param_list = []
        self.energy_list = []
        callback = partial(store_process, self.param_list, self.energy_list)
        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=self.n_layers,
            initial_point=[
                random.random() * 2 * np.pi for i in range(2 * self.n_layers)
            ],
            callback=callback,
        )
        op, _ = Q_to_paulis(self.Q)
        result = qaoa.compute_minimum_eigenvalue(op)
        return result

    def converge_distribution(self, dim, n_layers, n_cores=8):
        from multiprocessing import Pool

        backend = Aer.get_backend("aer_simulator")
        self.n_layers = n_layers

        def find_converge(dim):
            self.Q = symmetric_matrix_generator(dim)
            self.default_QAOA(backend)
            lowest = abs(np.mean(self.energy_list[-20:-1]))
            return np.where(self.energy_list < 0.99 * lowest)

        # with Pool(n_cores) as p:
        p = Pool(n_cores)
        distribution = p.map(find_converge, [dim for i in range(1000)])

        return distribution

    def portfolio_optimization(self, backend: Hardware_Backend) -> None:
        pass

    def total_benchmark(self, backend: Hardware_Backend) -> float:
        self.standard_QAOA(backend)
        self.portfolio_optimization(backend)

    def fit(data) -> (float, float):
        # fit data as a normal distribution
        def Gaussian_distr(x, sigma, mu):
            return (
                1
                / (sigma * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * (((x - mu) / sigma) ** 2))
            )

        upper = int(np.ceil(np.max(data)))
        lower = int(np.floor(np.min(data)))
        hist_y, hist_x = np.histogram(
            data, np.linspace(lower, upper, upper - lower + 1)
        )
        hist_x = hist_x[:-1]
        hist_y = hist_y / np.shape(data)[0]  # normalization
        [sigma, mu], _ = curve_fit(
            Gaussian_distr, hist_x, hist_y, p0=[1, np.average(data)]
        )
        residuals = hist_y - Gaussian_distr(hist_x, sigma, mu)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((hist_y - np.mean(hist_y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print("r_squared", r_squared)
        plt.plot(hist_x, hist_y)
        plt.plot(hist_x, Gaussian_distr(hist_x, sigma, mu))
        return sigma, mu
    
    def get_accuracy(self, backend, options=None):
        # return the probability of measuring the correct result
        gs = ground_state(self.Q)
        result = self.default_QAOA(backend, options=options)
        return result.eigenstate[gs]
    
    def get_convergence(self, backend, options=None):
        # return the number of steps with which QAOA converge
        self.default_QAOA(backend, options=options) # no need to know the result, callback recorded everything
        lowest = np.mean(self.energy_list[-20:-1])
        return np.where(self.energy_list < 0.99 * lowest)[0][0]
    
    def get_approx_rate(dim, n_layers):
        rate = 0
        return rate

    def save_result(data, file_name, header_name) -> None:
        # save the result to a csv file
        try:
            df = pd.read_csv(file_name)
            df[header_name] = data
        except:  # in case this is a new file
            df = pd.DataFrame(data)
            df.columns = [header_name]
        df.to_csv(file_name, header=True, index=False)
