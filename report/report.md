---
title: |
    COMP36212 Assignment EX3.
    Gradient-Based Optimisation

author: Vladislav Yotkov
bibliography: custom.bib
header-includes:
- \usepackage{acl}
- \usepackage{natbib}
- \usepackage[inline]{enumitem}
- \usepackage[justification=centering]{caption}
- \bibliographystyle{acl_natbib.bst}
- \setcounter{secnumdepth}{5}
- \usepackage{algorithm} 
- \usepackage{algorithmicx} 
- \usepackage{algpseudocode}
graphics: yes
fontsize: 10pt
geometry:
- top=20mm
- bottom=20mm
- left=20mm
- right=20mm
---

# Problem Statement {#sec:problem-statement}
In this assignment we explore the optimisation methods of an Artificial Neural Network (ANN) for image classification. 
The task is to train a multi-layer perceptron, applied to classify images of handwritten digits (0-9) from the MNIST dataset [@lecun98mnist], 
aiming to achieve maximum classification accuracy. The process involves the computation of a prediction from a sample input (i.e., an image), the 
comparison of the network prediction with the true label to formulate an objective function for optimisation (cross-entropy loss, Eq. \ref{eq:cross-entropy}), 
and the use of target optimisation approaches to minimise the objective function with respect to the network parameters.
\begin{equation}
    \label{eq:cross-entropy}
    L = -\sum^{N}_{k} \hat{y}_k \log(P_k)
\end{equation}

## Dataset {#sec:dataset}
The MNIST dataset [@lecun98mnist] is a collection of 70,000 images of handwritten digits (0-9), each of which is a 28x28 pixel image, with pixel intensity values ranging from 0-255.
For input into the ANN, it is divided into 60,000 training and 10,000 testing images, 
while each sample is flattened into an array of 784 elements, which is further normalised to the range $0 \leq x_i \leq 1$.

## Artificial Neural Network (ANN) {#sec:ann}
The ANN is a multi-layer perceptron, consisting of five layers with a total of 784 neurons in the input layer, 
three hidden layers containing 300, 100, and 100 neurons respectively, and an output layer of 10 neurons (Figure \ref{fig:ann})
for a total of $276,200$ trainable parameters.
\begin{figure}[h]
    \centering
    \includegraphics[width=0.5\textwidth]{charts/ann.png}
    \caption{Artificial Neural Network (ANN) architecture, taken from the assignment specification.}
    \label{fig:ann}
\end{figure}
The activation function used for the hidden layers is the Rectified Linear Unit (ReLU, Eq. \ref{eq:relu}),
while the output layer uses the softmax function (Eq. \ref{eq:softmax}), which normalises the output logits to a probability distribution over the 10 classes.
\begin{equation}
    \label{eq:relu}
    ReLU(x) = \max(0, x)
\end{equation}
\begin{equation}
    \label{eq:softmax}
    softmax(x)_i = \frac{e^{x_i}}{\sum^{N}_{j=1} e^{x_j}}
\end{equation}

# Stochastic Gradient Descent (SGD) {#sec:sgd}
Stochastic Gradient Descent is one of the most significant and widely used optimisation algorithms used in machine learning.
It is a variant of the Gradient Descent (GD) algorithm, which is used to find the local minimum of a function by iteratively moving in the direction of the negative gradient.
However, the main difference is that SGD uses a random sample of the training data in order to compute the gradient, unlike the
GD algorithm which is applied over the entire dataset - leading to high computational costs.
In our work, we will be exploring the **on-line** (i.e., stochastic) and mini-batch SGD (where $m \ll N$, such that $m$ is the 
size of the mini-batch, and $N$ is the size of the training dataset) variants of the algorithm.
We can define more formally this optimisation method in Algorithm \ref{alg:sgd}, while the update rule is given by Eq. \ref{eq:sgd}.
\begin{equation}
    \label{eq:sgd}
    w = w - \frac{\eta}{m} \sum_{i=1}^{m} \nabla L_i(x_i)
\end{equation}

\begin{algorithm}[h]
    \caption{Mini-batch Stochastic Gradient Descent (SGD)}
    \begin{algorithmic}[1]
        \Require Training dataset $D$
        \Require Learning rate $\eta$ 
        \Require Mini-batch size $m$
        \Ensure Optimized model parameters $w$
        \State Initialize model parameters $w$
        \While{stopping criteria not met}
            \State Sample an $m$-sized mini-batch from $D$: $\{(x_1, y_1), (x_2, y_2), ..., (x_m, y_m)\}$
            \State Compute gradient: $\nabla L = \frac{1}{m} \sum_{i=1}^{m} \nabla L_i(x_i, y_i)$
            \State Update model weights: $w = w - \eta \nabla L$
        \EndWhile
        \State \Return Optimized model parameters $w$
    \end{algorithmic}
    \label{alg:sgd}
\end{algorithm}

We must further note a few key benefits of using an SGD over general GD:
1. On-line (i.e., stochastic) learning requires a smaller step size (i.e., learning rate) to counteract inherent noise, 
    resulting in smoother but slower adaptation [@wilson2003generalinnefficiencybatchtraining]. 
    Despite the increased runtime, small batches offer a beneficial regularization effect and over time, noise averages out, 
    driving the weights towards the true gradient [@Goodfellow-et-al-2016].


# Improving Convergence {#sec:improving-convergence}

# Adaptive Learning {#sec:adaptive-learning}

# Conclusion {#sec:conclusion}
