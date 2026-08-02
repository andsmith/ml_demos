


# NLP Experiment (core description and new tab description)
 I Used our vocabulary condenser and created a reduced Tiny Stories corpus in ./data. We need to create the NLP core for a new NLP tab in the Willshaw demo  (for a future tab named "Next word prediction")
 For this spec, you are loading in that tokenization exactly as the named_reduction_explorer created it.



## Core
 
Word modules implemented as Willshaw attractors can be wired together using lateral connections to simulate NLP.
For this demo we use up to 6 word modules to represent the first 6 words of a sentence.  
Each module has |D| neurons, |V| symbols, using (WLOG) the same vocabulary. 
Each word gets a random sparse encoding, sparsity determined optimally for the |D| and |V| to minimize expected neuron collisions.
The pattern of excitation on a module includes inputs from all other modules wired to it.

The pattern of activation on a module indicates all the symbols currently "active" in the module.  T
The job of the lateral connections to a module is to activate all symbols in that module that are associated with the symbols wired to it.
The job of the willshaw connections is to reduce the set of active symbols to the most excited one "locking it down" (or top N "paring it down").

Inference of a missing word happens in two phases:
1. compute lateral input from words that are present, expected to activate correct and spurious symbols on the module
2. Do willshaw iterations until a single symbol remains active on the unit.

The lateral connections between modules are organized into "knowledge bases", since they represent learned statistical relationships between words in a constant relative (token) distance in the sentence.

### inference
Processing can happen on two levels:
* Nerosimulation (low-level) - Using actual lateral binary connections between units (internally, representing / computing the binary activations from excitations of simulated neurons)
   1. lateral phase for getting excitation on unit U from locked-down neighbors N_i is summing up products of binary vectors on each N_i with the "knowledge-weight" matrix from N_i to U.
   2. Attrractor phase is willshaw circuit running to stability.
* Confabulation (high-level) - Using statistics of symbols that the neurons are approximating. (i.e. representing weighted lists of symbol activations on modules)
   1. Lateral phase for getting weighted list of symbol excitations.  For the word w active on unit N_i, the row of the "knowledge-base" matrix from N_i to U, row corresponding to w is the vector of probabilities that each word was active in U given word w was active in N_i.  The probability vectors are multiplied element-wise (or log-added) for all locked-down neighbors of U to obtain the excition distribution on words in U.
   2. Attractor phase is simply keeping the top suported word, or top N words. 
## your task - stand-alone nlp functions that can be used in the demo

For this version, we are only implementing high-level confabulation, but we are visually depicting the neurosimulation using module_art, word lists with weights, colors, etc.
This means you don't need to compute unit-unit knowledge-weight matrices based on the word (aka token, aka symbol) encoding, just the knowledge-base matrices based on statistics from the corpus.

Write extract_knowledge_base.py to learn the probability tables, it takes as arguments an exhaustive list of all the KBs to extract (in the form of int pairs:  1-6 2-6 3-6 4-6 5-6 learns five KBs that can be used to predict the 6th word in a sentence.  It can also extract relative KBs for "sliding-window" learning (relative not fixed positions within the sentence). 
Both kinds of KBs get saved to dir ./knowledge_bases as npz files with a different filename to indicate which ind. 
Write extract_tiny_kbs.sh to make the calls to do this for absolute KBs (All KBs from position i to position j, where 1<=i<6 and j>i) and the same for relative KBs, so we can generate up to the 6th word in the beginning of a sentence or the next word from the previous 5 mid-sentence.

Write confabulation_high_level.py, that does 6-th word inference (absolute or relative) given an input of 5 words.