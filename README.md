# HamilToniQ

Developed by: Felix Xu, Louis Chen

Table of Contents:

1. [Introduction](#introduction)
2. [Quick Start](#quickstart)
3. [Our Result](#ourresult)

<a name="introduction"></a>

## Introduction 

A benchmarking toolkit designed for QAOA performance on real quantum hardware.

Using HamilToniQ, you can

* benchmark a quantum processor.
* compare the quantum processors and find the one most suitable for your case.

<a name="quickstart"></a>

## Quick Start 

### Installation

Install the *HamilToniQ* toolkit by run the following code in the Terminal.

```shell
cd /path/to/your/directory
git clone https://github.com/FelixXu35/hamiltoniq.git
cd hamiltoniq
pip install -e .
pip install -r requirements-dev.txt
```



### Bechmark a backend

Simply copy and run the following python code:

```python
from hamiltoniq.bechmarking import Toniq

toniq = Toniq()
backend = <your_backend>
dim = <your_prefered_dim>
n_layers = <your_prefered_n_layers>
n_cores = <number_of_cores_in_your_PC>

score = tonic.run(backend=backend, dim=dim, n_layers=n_layers, n_cores=n_cores)
```

<a name="ourresult"></a>

## H-Scores

The following results were obtained on the built-in Q matrices and sorted by the scores with `n_layers=1`.

Note that comparsion across different number of qubits is meaningless!

##### 3 qubits

<p align=center><img src="./hamiltoniq/H_Scores/qubit_3.png" alt="n_qubits=3" width="700" /></p>

##### 4 qubits

<p align=center><img src="./hamiltoniq/H_Scores/qubit_4.png" alt="n_qubits=4" width="700" /></p>

##### 5 qubits

<p align=center><img src="./hamiltoniq/H_Scores/qubit_5.png" alt="n_qubits=4" width="700" /></p>

##### 6 qubits

<p align=center><img src="./hamiltoniq/H_Scores/qubit_6.png" alt="n_qubits=4" width="700" /></p>

## Our Paper

 If you are interested in this research, please consider citing our paper:

