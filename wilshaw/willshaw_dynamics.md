# Willshaw Attractors

## Purpose

A **Willshaw attractor** is the computational primitive underlying the cortical "symbol" representation in Confabulation Theory. Every cortical module contains one such attractor network.

Its purpose is to perform **content-addressable memory** (CAM):

> Given an incomplete, noisy, or partially incorrect activation pattern, converge to the unique stored pattern that best matches it.

Unlike a conventional lookup table, the stored memories are **not indexed by address**. Instead, the memory is retrieved by similarity between the current activity pattern and previously learned patterns.

This makes a Willshaw attractor behave much more like human recall:

- partial cue → complete memory
- noisy cue → corrected memory
- missing features → reconstructed features
- order independent

The attractor therefore implements **pattern completion**.

## Biological Interpretation

In Confabulation Theory each cortical module represents one "lexicon."

Examples:

- visual objects
- words
- phonemes
- faces
- colors
- spatial locations

Each module contains approximately:

- tens of thousands to millions of neurons
- sparse excitatory recurrent connections
- inhibitory competition
- recurrent settling dynamics

The exact biological implementation is not important for a software demonstration.

Instead, the module can be modeled as a recurrent binary associative memory.

## Symbols

A **symbol** is **not** a single neuron.

Instead it is represented by a stable distributed pattern of activity across many neurons.

Typical sparsity is roughly 1–5% active neurons, with the remaining neurons inactive. Sparse representations dramatically increase memory capacity.

## Learning

Let x be a binary activity vector representing one stored symbol.

For every pair of simultaneously active neurons:

```text
if x_i = 1 and x_j = 1:
    W[i,j] = 1
```

This is a binary Hebbian learning rule.

Weights are permanent binary values:

- 0 = no learned association
- 1 = learned association

No gradient descent, optimization, or backpropagation is involved.

## Mathematical Form

The learned recurrent matrix is the binary OR of outer products over all stored patterns.

Equivalent implementation:

```text
for each stored pattern:
    for every active neuron pair:
        W[i,j] = 1
```

The matrix is symmetric and the diagonal is usually ignored.

## Pattern Completion

Later, when a partial pattern is presented, recurrent excitation reconstructs the missing neurons.

Suppose a stored symbol activates neurons:

```text
A B C D E F
```

The cue contains only:

```text
A B _ D _ F
```

After recurrent updates:

```text
A B C D E F
```

The network has completed the pattern.

## Attractor Dynamics

Let a(t) be the current binary activity vector.

Each iteration computes recurrent support:

```
input = W · a(t)
```

Each neuron counts how many currently active neurons support it.

A simplified update rule is:

```
a_i(t+1) = 1 if input_i > threshold
           0 otherwise
```

Iterations continue until the activity vector stops changing.

At convergence:

```
a(t+1) = a(t)
```

The network has reached a stable attractor.

## Energy View

Each stored pattern forms a basin of attraction.

Nearby noisy patterns converge toward the stored memory through repeated recurrent updates.

This provides robust error correction and pattern completion.

## Why Binary Weights Work

Sparse representations activate only a tiny fraction of neurons.

Consequently only a tiny fraction of all possible neuron pairs become connected.

Even after many stored memories, accidental overlap remains relatively low, allowing surprisingly large storage capacity.

## Relation to Content-Addressable Memory

Traditional RAM retrieves data by address.

A Willshaw attractor retrieves data by similarity to stored contents.

The memory itself acts as the address.

## Difference from Hopfield Networks

| Hopfield | Willshaw |
|-----------|-----------|
| Analog or bipolar weights | Binary weights |
| Hebbian sums | Binary OR learning |
| Often dense coding | Extremely sparse coding |
| Lower sparse capacity | Very high sparse capacity |

## Role in Confabulation Theory

The Willshaw attractor does **not** perform reasoning.

Its sole function is to convert a noisy or incomplete activation pattern into the nearest valid stored cortical symbol.

Higher-level reasoning occurs through knowledge links between cortical modules. Those links bias the attractor toward candidate symbols, while the attractor itself performs local cleanup until a stable symbol emerges.

## Implications for a Demo

A faithful implementation requires only:

- sparse binary activity vectors
- a binary symmetric recurrent weight matrix
- Willshaw learning
- iterative thresholded recurrent updates
- convergence detection

Everything else in Confabulation Theory builds upon this primitive.
