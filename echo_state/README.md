# Echo State Networks

[Echo State Networks](http://www.scholarpedia.org/article/Echo_state_network) (ESNs) are an example of "[reservoir computing](https://en.wikipedia.org/wiki/Reservoir_computing)".  A complex dynamical system is used to model timeseries data by driving it with a sequence of inputs, collecting the sequence of internal states that evolve over time, then mapping the sequence of (input, state) pairs to the correct target output sequence.  The input and internal weights are random and fixed.   Only the output layer is learned.

Let the timeseries learning task be defined by:
* $X(t)$ the sequence of input vectors, and
* $Y(t)$ the sequence of output target vectors.

Then at time $t$ the ESN units are defined to have excitation, activation/state, and output:
* Excitation:  $E(t) = W_{res} s(t-1) + W_{in} X(t) + W_{f} O(t-1)$    
* Activation:  $S(t) = tanh(E(t))$
* Output: $O(t) = W_{out} S(t)$

where

* $W_{res}$ is the matrix of reservoir weights
* $W_{in}$ is the matrix of input weights
* $W_{out}$ is the matrix of output weights
* $W_{f}$ is the matrix of feedbaclk weights (optionally zero).


The "state" of the network at time $t$ refers to the vector of activations, $S(t)$.


For more info see [Scholarpedia](http://www.scholarpedia.org/article/Echo_state_network).


## Transforming signals with an ESN:

Task:  Given signal $f(x)$ as input, produce $g(x)$ as output (aka, signal transduction).

![waves](/echo_state/assets/waves.png)
Given X(t) is a square wave, can we generate the other three, i.e. can we find an output matrix that maps the state, driven by $X(t)$ to the output value $Y(t)$?

To test the ability of various sized ESNs on this taks, we generate an input and output signals with 50,000 samples, run the input through a randomly initialized ESN to collect the internal states $S(t)$, and then find the output weights $W_{out}$ using least squares.

Run `>python esn_tester.py [size]` to create an Echo State Network with `size` internal units:

![fixed_points](/echo_state/assets/signals.png)
This will train three networks using the square wave as input signals (top row) and the other types of waves as the desired targets.  The training data and training output are shown in the left column.  Testing data is training data with a given phase shift (right columns).  The dashed line shows the 'washout' parameter, i.e. output weights are learned using only the input/state pairs generated after the washout iteration (and the reported error is calculated using the same samples).

TODO:  Experiment with frequency generalization as well as phase.


## How long does it take an ESN(n) to settle?

To ensure $S(0)$ is the same at the start of every input sequence, it is first run with input=0 until it converges (which is guaranteed by the "echo state property").  What isn't clear is how long this will take.  

#### As a function of reservoir-size and spectral-radius

To explore this, run `> python find_equilibrium.py` to generate the following plot:


![fixed_points](/echo_state/assets/fixed_points_b.png)


This shows `spectral_radius < 1` means the network always converges (tested out to `max_iter`), and that the average settling time when running the network with a vector of zeros as the input is more a property of spectral radius than the reservoir size.  Increasing reservoir sizes seems to decrease the variance on settling time.

#### As a function of sparsity:

The script will also show the mean settling time for different sized networks with different sparsity parameters, $s  \in [0,1)$, representing the fraction of internal (Res-res) weights that are set to zero after initializing.

![sparsity](/echo_state/assets/sparsity_and_convergence.png)

This creates n=50 random ESNs of the different sizes and sparsity constraints, then computes their settling time (Run with zero inputs until state variables don't change more than 1e-7, or similar.)

The mean settling time appears constant wrt. reservoir size and sparsity, however increasing sparsity seems to increase the variance.