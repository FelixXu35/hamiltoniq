# HamilToniQ

[TOC]

A benchmarking toolkit designed for QAOA performance on real quantum hardware.

Using HamilToniQ, you can

* use built-in reference cases to benchmark an actual quantum processor.

* build your own reference cases.

## Concept

What does a user value?

accuracy, the possiblility of finding the correct answer in one execuation of QAOA.

What is the criatira of a good benchmarking tool?

straightforward - use only one number to tell you the overall performance

low variance - in the benchmarking of with a possibility density distribution, the influence of the variance should be kept to a very low level



## Our Result

Since the results depend on the instance/Q matrix used in the benchmarking process, we do not resommand comparing the score across different dimensions.

### the number of qubits is 3

![dim_3](./HamilToniQ/our_results/dim_3.png)

### the number of qubits is 4

![dim_4](./HamilToniQ/our_results/dim_4.png)