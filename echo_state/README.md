# Echo State Networks

[Echo State Networks](http://www.scholarpedia.org/article/Echo_state_network) (ESNs) are an example of "[reservoir computing](https://en.wikipedia.org/wiki/Reservoir_computing)", where a complex but fixed dynamical system is driven by a sequence of inputs and a map is learned from the sequence of dynamic states to the desired outputs.

For more info see [Scholarpedia](http://www.scholarpedia.org/article/Echo_state_network).


## Transforming signals with an ESN:

Task:  Given signal $f(x)$ as input, produce $g(x)$ as output (aka, signal transduction).

![waves](/echo_state/assets/waves.png)
Given f(x) is a square wave, can we generate the other three, i.e. can we find an output matrix that maps the state, driven by $f(x)$ to the output value $g(x)$.

Run `>python esn_tester.py [size]` to create an Echo State Network with `size` internal units:

![fixed_points](/echo_state/assets/signals.png)

This will train three networks using the square wave as input signals (top row) and the other types of waves as the desired targets.  The training data and training output are shown in the left column.  Testing data is training data with a given phase shift (right columns).  The dashed line shows the 'washout' parameter, i.e. output weights are learned using only the input/state pairs generated after the washout iteration (and the reported error is calculated using the same samples).

TODO:  Experiment with frequency generalization as well as phase.


## How long does it take an ESN(n) to settle?

run `> python find_equilibrium.py` to generate the following plot:


![fixed_points](/echo_state/assets/fixed_points.png)

This shows `spectral_radius < 1` means the network always converges, and that the average settling time when running the network with a vector of zeros as the input is more a property of spectral radius than the reservoir size.
