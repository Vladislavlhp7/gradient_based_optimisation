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
- \usepackage{subcaption}
- \usepackage{caption}
graphics: yes
fontsize: 12pt
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
In our work, we will be exploring the **on-line** (i.e., stochastic) and **mini-batch SGD** (where $m \ll N$, such that $m$ is the 
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
2. Large batch sizes exhibit a degradation in quality [@mishkin2017systematicconvolutionalneuralnetwork] and a low generalisation capability and are 
    prone to getting stuck in local minima due to their convergence to sharp minimizers of the training function [@keskar2017largebatch].

## Configuration Experiments {#sec:config-experiments}
We experimented with different hyperparameters for the SGD algorithm, in order to find the optimal configuration for our ANN classifier:

1.**Learning rate**: The learning rate $\eta$ is the step size of the gradient descent algorithm, and is the most important hyper-parameter [@Goodfellow-et-al-2016].
    Setting it to larger values risks rapid changes and overshooting the minimum, while converging to a suboptimal solution.
    On the other hand, a smaller learning rate requires more training epochs to converge, and is more likely to get stuck in a local minimum.
    In our work on SGD, we experimented with the following values $\eta = \{0.1, 0.01, 0.001\}$.

2.**Batch size**: The batch size $m$ is the number of training samples used to compute the gradient at each iteration.
    The mini-batch SGD is said to follow the gradient of the true generalization error [@Goodfellow-et-al-2016]
    by computing an unbiased estimate (implying data is sampled randomly).
    When $m = 1$, the algorithm is called **on-line** (i.e., stochastic) learning, whereas for $m = N$ it is called **batch** learning.
    In our work, we experimented with the following batch size values $m = \{1, 10, 100\}$.

As requested, we track the convergence of the training process by plotting the loss and test accuracy over 
all epochs for learning rates $\eta = 0.1$ and batch sizes $m = 10$ in Fig. \ref{fig: 1.3}. 
We can see that there is a **rapid fall in the loss** function, and a corresponding **jump in the test accuracy** 
in the first epoch, which is caused by the large learning rate [@Goodfellow-et-al-2016]. 
Furthermore, the training process is **stable** with a monotonically decreasing loss function, which demonstrates the theoretical 
correctness of the SGD implementation. Regarding the **generalization capability** of the model, we can see that the test accuracy
**does not suffer from over-fitting**, as it is constantly increasing over the epochs, and reaches a value of $98$% at the last (i.e., tenth) epoch.

\begin{figure}[h]
    \centering
    \includegraphics[width=0.5\textwidth]{charts/1.3.png}
    \caption{Loss and test accuracy for lr=0.1 and $m = 10$.}
    \label{fig: 1.3}
\end{figure}

Additionally, while this hyperparameter configuration somewhat **exhibits a zig-zagging behaviour** from Epoch 1 onwards, which is
common to stochastic GD approaches [@wilson2003generalinnefficiencybatchtraining], the oscillations are limited in magnitude,
but we cannot yet confirm if the model has reached the global optimum.
To address this, we rerun this experiment over 20 epochs (Fig \ref{fig: oscillations}) and find that the model does not further improve the performance,
while the zig-zagging effect is now not only present but also more pronounced between the 10th and 20th epochs - implying that 
the step size is too large to find the global optimum.

\begin{figure}[h]
\centering
\includegraphics[width=0.5\textwidth]{charts/oscillations.png}
\caption{Test accuracy for lr=0.1 and $m = 10$ over 20 epochs.}
\label{fig: oscillations}
\end{figure}

From the experiments on both learning rates and batch sizes we reach the following conclusions:

1. **Instability for $lr=0.1, m=1$**: From Fig. \ref{fig: batch-size-1-acc} it becomes clear that a combination of
    a large learning rate and a small batch size leads to a very unstable optimization process, with a nan loss value and 
    a constant test accuracy of $0.098$ (i.e., random guessing) as visible in Table \ref{tab:sgd}. These results are expected due to 
    the erratic stochastic updates, and the high variance in the gradient estimates [@Goodfellow-et-al-2016]. In contrast,
    just by decreasing the learning rate to $\eta = 0.01$ or by increasing the batch size to $m = 10$ we can achieve a stable and 
    an accurate solution ($\geq 95$%).
2. **Slow convergence for large $m$**: While a larger batch size leads to a more stable learning because of the averaging effect,
    it can suffer from a lack of generalization caused by the convergence to sharp minimizers and the inability to escape them post factum [@keskar2017largebatch].
    This is evident for $(lr=0.001, m=10)$ and $(lr=0.001, m=100)$ from Fig. \ref{fig: batch-size-10-acc} and \ref{fig: batch-size-100-acc}, respectively,
    where the test accuracy convergence slows down over all 10 epochs.
3. **Striking a good balance**: The optimal SGD solution depends on the trade-off between the learning rate and the batch size.
    However, the following configurations: $(lr=0.1, m=100)$, $(lr=0.01, m=10)$, and $(lr=0.001, m=1)$ seem to strike a 
    good balance between the two hyperparameters, achieving low average loss and high test accuracy. 
    It is also not unreasonable to claim that the learning rate and the batch size seem inversely correlated in terms of model performance.


## Analytical Validation {#sec:analytical-validation}
To validate the correctness of the provided analytical gradient calculation, we approximate the derivative using 
three different finite-difference methods [@ford2015numerical]:

1. Forward difference: $\frac{f(x + h) - f(x)}{h}$
2. Backward difference: $\frac{f(x) - f(x - h)}{h}$
3. Central difference: $\frac{f(x + h) - f(x - h)}{2h}$

where we set the step size $h$ to $10^{-8}$.

Due to the significant computational cost of the all these methods (each takes ~1s per sample), 
we only compute the numerical gradients for a single sample at the end of the first training epoch and compare them 
to the analytical ones.
We carry out our experiments using a learning rate of $\eta = 0.1$ and a mini-batch size of $m = 10$.
The results shown in Figure \ref{fig:mean_rel_error} and Figure \ref{fig:rel_dist} indicate that the analytical and 
numerical gradients differ by less than $3 \times 10^{-4}$% (central difference) on average, which goes to show that the
analytical gradient calculation is correct.
As expected the central difference method provides a better approximation and a lower truncation error than the forward
and backward difference ones due to the fact that it is second-order accurate, while the others are first-order accurate.
This is clearly visible from the lower mean relative error and the narrower distribution of the central difference method.
And while these results demonstrate our trust in the analytical solution, we note that the numerical algorithms are too 
computationally expensive to be used in practice over the range of all training data.

\begin{figure}[h]
    \centering
    \includegraphics[width=0.5\textwidth]{charts/mean_rel_error.png}
    \caption{Mean relative error of the finite-difference methods.}
    \label{fig:mean_rel_error}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.5\textwidth]{charts/rel_dist.png}
    \caption{Relative error distribution of the finite-difference methods.}
    \label{fig:rel_dist}
\end{figure}

# Improving Convergence {#sec:improving-convergence}
In order to counteract the zig-zagging effect of the large learning rate, but still maintain fast convergence, we will explore
two different techniques: learning rate decay and momentum.

## Learning Rate Decay {#sec:learning-rate-decay}
Learning rate decay is a common technique often useful in practice [@wilson2003generalinnefficiencybatchtraining] that 
makes more aggressive steps at the beginning and then gradually decrease the learning rate over time, 
allowing the model to make larger updates initially, followed by smaller and more fine-grained ones towards the end.
More formally, we define the learning rate decay as follows:
\begin{equation}
    \eta_{k} = \eta_{0}(1 - \alpha) + \alpha \eta_{N}, \quad \text{where} \quad \alpha = \frac{k}{N}
    \label{eq:learning-rate-decay}
\end{equation}
where $\eta_{0}$ is the initial learning rate, $\eta_{N}$ is the final learning rate, $k$ is the current epoch, and $N$ is the total number of epochs.

Some researchers even claim that using learning rate decay is equivalent to increasing the batch size [@smith2018dontdecay]
in terms of model performance, where the latter leads to significantly fewer parameter updates
(i.e., improved parallelism and shorter training times).

Nevertheless, throughout our experiments we will fix the initial and the final learning rates to $\eta_{0} = 0.1$ and $\eta_{N} = 0.001$ respectively,
and we will vary the number of batches per epoch $m$ from 1 to 10, and to 100.
We argue that it is reasonable to use a more aggressive learning rate at the beginning of the training process to explore the parameter space, 
and then slowly calibrate it so as not to overshoot the optimal solution which was the case for SGD in Section \ref{sec:config-experiments}.

The results shown in Table \ref{tab:sgd_decay} indicate that:

1. the decay technique improves the final average loss and test accuracy for $m=10$ because of the aggressive initial learning rate and the batch averaging effect;
2. however, it also suffers from the SGD's shortcomings from Section \ref{sec:config-experiments} when $m=1$ (i.e., on-line learning), 
   which we believe is once again caused by the erratic high-variance gradient updates.

\begin{table}[ht]
    \centering
    \begin{tabular}{|c|c|c|c|c|}
        \hline
        Avg Loss & Test Acc & $\eta_{0}$ & $\eta_{N}$ & m\\
        \hline
            2.303 & 0.098 & 0.1 & 0.001 & 1\\
            \textbf{0.002} & \textbf{0.983} & 0.1 & 0.001 & 10\\
            0.031 & 0.977 & 0.1 & 0.001 & 100\\
        \hline
    \end{tabular}\label{tab:sgd_decay}
    \caption{SGD performance with learning rate decay: Loss and Accuracy}
\end{table}

We further plot the loss and test accuracy for the best batch size (i.e., $m=10$) in Figure \ref{fig:decay_10}
to demonstrate the effect of the learning rate decay on the zigzag effect common to SGD [@wilson2003generalinnefficiencybatchtraining].
We observe that this technique indeed helps the model converge faster and more smoothly to the optimal solution.

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.49\textwidth]{charts/decay_10.png}
    \caption{SGD with learning rate decay: Loss and Accuracy}
    \label{fig:decay_10}
\end{figure}

## Momentum {#sec:momentum}


# Adaptive Learning {#sec:adaptive-learning}
In this section we will explore the AdaGrad optimizer [@hoffer2017adagrad], defined in Algorithm \ref{alg:adagrad}.
The main innovation of this technique is that it introduces individual adaptive learning rates ($\eta$) for each model weight, 
whereas before $\eta$ was a global hyperparameter set uniformly for all parameters. 
This is possible through the accumulation of squared historical gradients per parameter $w_i$ over time, 
which are then used for the scaling of $w_i$'s learning rate; more formally it can be described as follows:

\begin{algorithm}[h]
    \caption{AdaGrad}
    \begin{algorithmic}[1]
        \Require Learning rate $\eta$
        \Require Initial model parameters $w$
        \Require Small constant $\epsilon$ (e.g., $10^{-8}$)
        \Ensure Optimized model parameters $w$
        \State Initialize $G_0$ as an empty diagonal matrix of the same size as $w$
        \While{stopping criteria not met}
        \State Compute gradient $g_t = \nabla J(w)$ at current parameters $w$
        \State Accumulate squared gradient: $G_t = G_{t-1} + g_t^2$
        \State Compute update: $\Delta w = - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$
        \State Update parameters: $w = w + \Delta w$
        \EndWhile
        \State \Return Optimized model parameters $w$
    \end{algorithmic}
    \label{alg:adagrad}
\end{algorithm}
where $\odot$ denotes the element-wise product of the two vectors, while $\nabla J(w)$ is the gradient of the loss function
$J(w)$ with respect to the model parameters $w$. The $\epsilon$ (usually set to $10^{-8}$) term is further added to avoid division by zero.

We further note three key aspects of AdaGrad:

1. **Parameter-Specific Learning Rates**: AdaGrad adapts the learning rate so that infrequent parameters (i.e., rare features)
   receive larger updates, whereas the frequent ones (i.e., common features) receive smaller updates.
   This is especially useful in computer vision tasks (e.g., image classification) with frequent or repetitive image patterns (e.g., background, main body of digit).
2. **No Manual Learning Rate Tuning**: AdaGrad eliminates the need to manually tune the learning rate by
   allowing it to adaptively tune for each parameter during training.
3. **Rapid Learning Rate Decay**: One issue with AdaGrad is that the learning rate monotonically decreases during training,
   which may cause premature convergence. To address this, two algorithm extensions were proposed, namely: RMSProp [@tieleman2012lecture] and Adam [@kingma2017adam], that
   attempt to resolve this issue by introducing a decaying average of the historical squared gradients $E[g^2]_t$.



# Conclusion {#sec:conclusion}

\begin{figure}[h]
    \centering
    \begin{subfigure}[b]{0.5\textwidth}
    \caption{Batch size (m) = 1}
    \includegraphics[width=\textwidth]{charts/acc_sgd_m=1.png}
    \label{fig: batch-size-1-acc}
    \end{subfigure}
    \vfill
    \begin{subfigure}[b]{0.5\textwidth}
    \caption{Batch size (m) = 10}
    \includegraphics[width=\textwidth]{charts/acc_sgd_m=10.png}
    \label{fig: batch-size-10-acc}
    \end{subfigure}
    \vfill
    \begin{subfigure}[b]{0.5\textwidth}
    \caption{Batch size (m) = 100}
    \includegraphics[width=\textwidth]{charts/acc_sgd_m=100.png}
    \label{fig: batch-size-100-acc}
    \end{subfigure}
    \caption{Test accuracy for different learning rates and batch sizes.}
    \label{fig: learning-rate-acc}
\end{figure}


\begin{table}[ht]
    \centering
    \begin{tabular}{|c|c|c|c|c|c|}
        \hline
        Avg Loss & Test Acc & $\eta$ & m & Time (s) \\
        \hline
        nan & 0.098 & 0.1 & 1 & 3533s \\
        0.025 & 0.977 & 0.1 & 10 & 3268s \\
        \textbf{0.017} & 0.976 & 0.1 & 100 & 3268s \\
        0.027 & 0.977 & 0.01 & 1 & 3557s \\
        \textbf{0.019} & 0.976 & 0.01 & 10 & 3338s \\
        0.17 & 0.950 & 0.01 & 100 & 3301s \\
        \textbf{0.019} & \textbf{0.978} & 0.001 & 1 & 3553s \\
        0.16 & 0.952 & 0.001 & 10 & 3332s \\
        0.53 & 0.872 & 0.001 & 100 & 3288s \\
        \hline
    \end{tabular}\label{tab:sgd}
    \caption{SGD performance: Loss and Accuracy}
\end{table}