# Threshold Selection in a Willshaw Attractor

This note derives practical threshold choices for recovering sparse binary patterns from a Willshaw associative memory.

## Setup

Let

- \(N\) = number of neurons (dimension)
- \(k\) = number of active neurons per stored symbol (sparsity)
- \(M\) = number of stored symbols (vocabulary size)

Each stored symbol is a binary vector with exactly \(k\) active neurons.

The Willshaw connectivity matrix is

\[
W_{ij} =
\bigvee_{\mu=1}^{M}
x_i^\mu x_j^\mu,
\]

where a connection exists if neurons \(i\) and \(j\) were co-active in any stored symbol.

---

## 1. Connection Density

A given neuron pair appears together in one random symbol with probability

\[
\frac{k(k-1)}{N(N-1)}
\approx
\left(\frac{k}{N}\right)^2.
\]

After storing \(M\) symbols, the probability that a connection exists is

\[
p
=
1-
\left(
1-\frac{k(k-1)}{N(N-1)}
\right)^M.
\]

For sparse codes,

\[
p
\approx
1-
e^{-Mk^2/N^2}.
\]

This is the density of ones in the Willshaw matrix.

---

## 2. Activation During Recall

Suppose the partial cue contains

\[
c
\]

correct active neurons.

Each neuron receives

\[
h_i
=
\sum_j W_{ij}x_j.
\]

### Correct neuron

Every cue neuron is connected to the remaining neurons in the stored pattern, so

\[
E[h_{\text{true}}]=c.
\]

### Incorrect neuron

Each cue neuron connects independently with probability \(p\), so

\[
E[h_{\text{false}}]=cp.
\]

The false activations are approximately distributed as

\[
h_{\text{false}}
\sim
\text{Binomial}(c,p),
\]

with

\[
\operatorname{Var}(h_{\text{false}})
=
cp(1-p).
\]

---

## 3. Choosing a Threshold

A useful threshold must satisfy

\[
cp
<
\theta
<
c.
\]

### Midpoint threshold

A simple choice is

\[
\boxed{
\theta
=
\frac{c+cp}{2}
=
c\frac{1+p}{2}
}
\]

which lies halfway between the expected true and false activations.

### Statistical threshold

A more conservative threshold is

\[
\boxed{
\theta
=
cp
+
\alpha
\sqrt{cp(1-p)}
}
\]

where

- \(\alpha=3\) gives few false positives,
- \(\alpha=4\) is conservative,
- \(\alpha=5\) is very conservative.

---

## 4. Capacity

Everything depends on

\[
\lambda
=
\frac{Mk^2}{N^2}.
\]

The connection density is

\[
p
=
1-e^{-\lambda}.
\]

When

\[
\lambda \ll 1,
\]

true and false activations are well separated.

When

\[
\lambda \approx 1,
\]

the matrix becomes saturated with ones, making reliable recovery impossible regardless of threshold.

---

## Example

Given

- \(N=1000\)
- \(k=20\)
- \(M=300\)

we obtain

\[
p
=
1-e^{-0.12}
=
0.113.
\]

If the cue contains

\[
c=10
\]

correct neurons,

then

- Correct neuron activation:

\[
E[h]=10.
\]

- False neuron activation:

\[
E[h]=1.13.
\]

A threshold around

\[
\theta\approx5
\]

cleanly separates the two populations.

---

## Iterative Recall

Each iteration of the Willshaw attractor consists of

1. Computing analog excitation

   \[
   h = Wx.
   \]

2. Thresholding

   \[
   x_i =
   \begin{cases}
   1,& h_i\ge\theta,\\
   0,&\text{otherwise}.
   \end{cases}
   \]

The resulting state is binary before the next iteration.

Many implementations replace the fixed threshold with winner-take-all inhibition, selecting the top \(k\) most excited neurons each iteration. This is equivalent to using an adaptive threshold that maintains constant sparsity.

---

## Practical Design Rule

1. Compute

   \[
   p = 1 - e^{-Mk^2/N^2}.
   \]

2. Estimate

   \[
   h_{\text{false}}
   \sim
   \text{Binomial}(c,p).
   \]

3. Choose a threshold above the upper tail of the false activation distribution (or equivalently use winner-take-all thresholding).

Reliable recall is only possible while the memory remains sufficiently sparse that \(p\) is well below 1.
