


# NLP Experiment (core description and new tab description)
 I Used our vocabulary condenser and created a reduced corpus in ./data. We need to create the NLP core for a new NLP tab in the Willshaw demo  (tab named "Next word prediction")

## Core
 
Word modules implemented as Willshaw attractors can be wired together using lateral connections to simulate NLP.  For this demo we use up to 6 word modules to represent the first 6 words of a sentence.  Each module has |D| neurons, |V| symbols, using (WLOG) the same vocabulary.  The lateral connections between modules are organized into "knowledge bases", since they represent learned statistical relationships between words. 

