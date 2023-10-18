# HamilToniQ

Developed by: Felix Xu, Louis Chen

Table of Contents:

1. [Introduction](#introduction)
2. [Quick Start](#quickstart)
3. [Our Result](#ourresult)
4. [Theory](#theory)

<a name="introduction"></a>

## Introduction 

A benchmarking toolkit designed for QAOA performance on real quantum hardware.

Using HamilToniQ, you can

* use built-in reference cases to benchmark an actual quantum processor.
* build your reference cases.

<a name="quickstart"></a>

## Quick Start 

### Installation

Install the *HamilToniQ* toolkit by run `pip install hamiltoniq` in the Terminal.

### Bechmark a backend

Simply copy and run the following python code:

```python
from HamilToniQ.bechmarking 
```



### Create a new reference



<a name="ourresult"></a>

## Our Result 

Since the results depend on the instance/Q matrix used in the benchmarking process, we do not resommand comparing the score across different dimensions.

##### the number of qubits is 3

<p align=center><img src="./HamilToniQ/our_results/dim_3.png" alt="n_qubits=3" width="500" /></p>

##### the number of qubits is 4

<p align=center><img src="./HamilToniQ/our_results/dim_4.png" alt="n_qubits=4" width="500" /></p>

<a name="theory"></a>

## Theory 

The idea of our toolkit comes from a few questions.

Q: What does a user value?

A: Accuracy, which is the possibility of finding the correct answer in one execution of QAOA.

Q: What are the criteria of a good benchmarking tool?

A: Easy to use - use only one number to tell you the overall performance of a backend with a specific number of qubits; Stable - in the benchmarking of with a possibility density distribution, the influence of the variance should be kept to a very low level.

Therefore, for each backend with a certain number of qubits, our toolkit gives a score, which indicates how much all kinds of noise influence the accuracy. We know that the accuracy comes as a distribution, so the score is based on the sampling in that distribution, and it is defined as:

$$
2\int_0^1F(x)g(x)dx\\
=\int^1_0\int^x_0 f(y)g(x)dxdy\\
or = \int_0^1F(x)\delta(x)dx
$$
where $g(x) = f(x) + \delta(x)$.
$$
\int_0^1\delta(x)=0
$$


The highest score should be the score of the ideal simulator, and this score is 1. Because
$$
\int_0^1F(x)g(x)dx = \left[F^2(x)\right]^1_0 - \int_0^1F(x)g(x)dx \to \int_0^1F(x)g(x)dx=0.5
$$
