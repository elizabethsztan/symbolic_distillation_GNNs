# Symbolic Distillation of Graph Neural Networks

## Description
The aim of this project is to reproduce the main results of the paper [Discovering Symbolic Models from Deep Learning with Inductive Biases](https://arxiv.org/abs/2006.11287) by Cranmer et al., as part of the MPhil in Data Intensive Science at the University of Cambridge. \
In this project, we train a different variations of a graph neural network (GNN) (standard, bottleneck, L1, KL and pruning) on particle datasets with different interaction forces (charge, $r^{-1}$, $r^{-2}$ and spring). In all cases, we train on 2D data with four interacting particles.\
We firstly validate whether the GNNs learn the true forces by performing a linear regression of the true forces on the most important messages. Seperately we then perform a symbolic regression to the message elements using *PySR* to test whether we are capable of reconstructing the true force laws. By combining deep learning with symbolic regression, this framework could be extended to search for new empirical laws in high-dimensional data.\
We have extended on the original project by introducing a new model variation, the pruning model, where the dimensionality of the messages decreases throughout training.\
We have also created a new demo Colab notebook, $\texttt{demo.ipynb}$, where the user can test the pipeline on any of the interaction forces or model variations, as an attempt to increase the reproducibility of the pipeline. 

