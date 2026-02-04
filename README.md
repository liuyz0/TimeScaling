# Softmax and Cross-entropy Lead to a Universal One-third Time Scaling of Loss when Learning Peaked Distributions

This repository contains code to reproduce the experiments in the paper [Universal One-third Time Scaling in Learning Peaked Distributions](https://arxiv.org/abs/2602.03685)

## Overview

<p align="center" width="100%">
<img src="./figures/Fig-0-0.png" alt="Alt Text" style="width:100%; min-width: 200px; display: block; margin: auto;">
</p>

Training large language models (LLMs) is computationally expensive, partly because the loss exhibits slow power-law convergence whose origin remains debatable. 

Through systematic analysis of toy models and empirical evaluation of LLMs, we show that this behavior can arise intrinsically from the use of softmax and cross-entropy. When learning peaked probability distributions, e.g., next-token distributions, these components yield power-law vanishing losses and gradients, creating a fundamental optimization bottleneck. This ultimately leads to power-law time scaling of the loss with a universal exponent of $1/3$. 

An interesting analogy can be drawn to statistical physics, where peaked Boltzmann distributions require low temperatures, leading to power-law free energy and internal energy (with inverse temperature). 

Our results provide a mechanistic explanation for observed neural scaling and suggest new directions for improving LLM training efficiency.

## Experiment-code Correspondence

All toy model experiments can be found in folder [./exp](./exp), where the names of scripts correspond to the type of experiment being run and are explained in Appendix of the paper.

LLM experiments are in [./LLMs](./LLMs) folder, we use the script for [Pythia-12B](./LLMs/pythia-logit-1.py) as an example. The scripts for other LLMs are similar. The code for fitting and analysis is in the [notebook](./LLMs/pythia-logit-0to3.ipynb).

|Experiment| Where in [Paper](https://arxiv.org/abs/2602.03685) | Code |
|--|--|--|
|Adam Scanning Temperatures and Learning Rates|Figure 1 and 4| [exp-1](./exp/exp-1.py)|
|Adam Scanning Temperatures|Figure 2| [exp-0](./exp/exp-0.py)|
|Numerics of Theory| Figure 3 |[test-2](./tests/test-2.ipynb)|
|Adam Scanning Initialization Scales| Figure 5, a and b | [exp-2](./exp/exp-2.py)|
|Adam with Weight Decay| Figure 5, c and d | [exp-3-2](./exp/exp-3-2.py)|
|LLM Temperature | Figure 6a | [test-6](./tests/test-6.ipynb)|
|LLM Evaluation | Figure 6, b and c | [LLMs folder](./LLMs/) |
|SGD Scanning Temperatures and Learning Rates| Appendix D.2 | [exp-1-1](./exp/exp-1-1.py)|
|Adam with Learning Rate Schedule| Appendix D.4 | [exp-2-1](./exp/exp-2-1.py)|
|Generalized Toy Model with Residual Layers| Appendix D.6 | [exp-9](./exp/exp-9.py)|

## Citation

```
@article{liu2026universal,
  title={Universal One-third Time Scaling in Learning Peaked Distributions},
  author={Liu, Yizhou and Liu, Ziming and Pehlevan, Cengiz and Gore, Jeff},
  journal={arXiv preprint arXiv:2602.03685},
  year={2026}
}
```

## Interested in Other Neural Scaling Laws?

- Width Scaling Due to Limited Representation: Superposition Yields Robust Neural Scaling ([paper link](https://arxiv.org/abs/2505.10465), [code link](https://github.com/liuyz0/SuperpositionScaling/tree/main))
