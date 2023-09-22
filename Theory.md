# Introduction

distribution of accuracy on simulator is $f(x)$, and its cumulative sum is $F(x)$.

distribution of accuracy on real quantum processor is $g(x)$.

# Proof

the score is
$$
2\int_0^1F(x)g(x)dx\\
=\int^1_0\int^x_0 f(y)g(x)dxdy\\
or = \int_0^1F(x)\delta(x)dx
$$
where $g(x) = f(x) + \delta(x)$.
$$
\int_0^1\delta(x)=0
$$


The highest score is 1. Since
$$
\int_0^1F(x)g(x)dx = \left[F^2(x)\right]^1_0 - \int_0^1F(x)g(x)dx \to \int_0^1F(x)g(x)dx=0.5
$$

		
