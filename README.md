[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)

# HamilToniQ: An Open-Source Benchmark Toolkit for Quantum Computers

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
pip install -r requirements.txt
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

## How to cite

If you used this package or framework for your research, please cite:

```text
@misc{xu2024hamiltoniq,
      title={HamilToniQ: An Open-Source Benchmark Toolkit for Quantum Computers}, 
      author={Xiaotian Xu and Kuan-Cheng Chen and Robert Wille},
      year={2024},
      eprint={2404.13971},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```

