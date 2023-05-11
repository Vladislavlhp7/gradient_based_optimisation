---
title: |
    COMP36212 Assignment EX3: Gradient-Based Optimisation

author: Vladislav Yotkov
bibliography: custom.bib
header-includes:
- \usepackage{acl}
- \usepackage{natbib}
- \usepackage[inline]{enumitem}
- \usepackage[justification=centering]{caption}
- \bibliographystyle{acl_natbib.bst}
- \setcounter{secnumdepth}{5}
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
three hidden layers containing 300, 100, and 100 neurons respectively, and an output layer of 10 neurons (Figure \ref{fig:ann}).
\begin{figure}[h]
    \centering
    \includegraphics[width=0.5\textwidth]{charts/ann.png}
    \caption{Artificial Neural Network (ANN) architecture, taken from the assignment specification.}
    \label{fig:ann}
\end{figure}
The activation function used for the hidden layers is the Rectified Linear Unit (ReLU, Eq. \ref{eq:relu}),
while the output layer uses the softmax function, which normalises the output logits to a probability distribution over the 10 classes.
\begin{equation}
    \label{eq:relu}
    ReLU(x) = \max(0, x)
\end{equation}
