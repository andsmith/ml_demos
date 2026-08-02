# TinyStories Semantic Tokenization Strategy for Confabulation Theory

## Goal

This specification describes a configurable semantic tokenization
pipeline for TinyStories. The primary objective is to **greedily
minimize vocabulary size while preserving the maximum amount of semantic
structure** useful for cortical-symbol simulations based on
Hecht-Nielsen's Confabulation Theory.

Unlike conventional tokenizers that optimize compression, this tokenizer
optimizes **semantic interpretability**. Every output symbol should
correspond to a concept that could plausibly be represented as a
cortical symbol.

## Guiding Strategy

The tokenizer performs a sequence of deterministic reductions. At every
stage, it greedily replaces multiple lexical forms with a single
semantic symbol whenever the distinction is not required by the
experiment.

The reduction should always favor:

1.  Lower vocabulary size.
2.  Higher semantic consistency.
3.  Human interpretability.
4.  Deterministic mappings.
5.  Configurable loss of detail.

The amount of abstraction should be controlled by configuration rather
than hard-coded rules.

------------------------------------------------------------------------

# Processing Pipeline

``` text
Raw text
    ↓
Sentence segmentation
    ↓
POS tagging
    ↓
Dependency parsing
    ↓
Named Entity Recognition
    ↓
Coreference Resolution
    ↓
Lemmatization
    ↓
Semantic normalization
    ↓
Ontology mapping
    ↓
Semantic symbol sequence
```

------------------------------------------------------------------------

# Greedy Named Entity Reduction

Proper names contribute almost nothing to story semantics while
dramatically increasing vocabulary.

The tokenizer should replace every named entity with a canonical
representative of its semantic archetype.

Example canonical names:

  Archetype           Canonical Name
  ------------------- ----------------
  GOOD_FEMALE_CHILD   Alice
  GOOD_MALE_CHILD     Bob
  GOOD_FEMALE_ADULT   Alice
  GOOD_MALE_ADULT     Bob
  BAD_FEMALE_ADULT    Eve
  BAD_MALE_ADULT      Mallory
  UNKNOWN_FEMALE      Alice
  UNKNOWN_MALE        Bob

Within a story:

Mary → Alice

Sarah → Alice

Emma → Alice

Tom → Bob

John → Bob

If multiple entities of the same archetype appear simultaneously, assign
deterministic numbered variants:

Alice Alice2 Alice3

Bob Bob2

The numbering is local to a single story.

The canonical names are merely readable placeholders. They should never
carry semantic information beyond the ontology itself.

The mapping table should be entirely user configurable.

------------------------------------------------------------------------

# Configurable Reduction Controls

The tokenizer should expose independent controls allowing systematic
exploration of the tradeoff between vocabulary size and semantic
fidelity.

## 1. Name Reduction

Options:

-   Preserve all names
-   Canonicalize names (default)
-   Replace names with ontology symbols only

Examples:

Mary

↓

Alice

↓

GOOD_FEMALE_CHILD

------------------------------------------------------------------------

## 2. Human Granularity

Possible levels:

Level 0

PERSON

Level 1

MALE FEMALE

Level 2

MALE_CHILD FEMALE_CHILD MALE_ADULT FEMALE_ADULT

Level 3

GOOD_MALE_CHILD GOOD_FEMALE_CHILD BAD_ADULT etc.

------------------------------------------------------------------------

## 3. Place Granularity

Examples

Level 0

PLACE

Level 1

INDOOR OUTDOOR

Level 2

HOME SCHOOL PARK STORE

Level 3

HOME_KITCHEN HOME_BEDROOM CLASSROOM

------------------------------------------------------------------------

## 4. Animal Granularity

Level 0

ANIMAL

Level 1

PET FARM_ANIMAL WILD_ANIMAL

Level 2

DOG CAT COW HORSE etc.

------------------------------------------------------------------------

## 5. Object Granularity

Examples

FOOD

↓

FRUIT

↓

APPLE

BALL

↓

TOY

↓

SOCCER_BALL

------------------------------------------------------------------------

## 6. Verb Reduction

Options

Only lemma

walked → WALK

Semantic grouping

walk run skip

↓

MOVE

said asked told

↓

SPEAK

------------------------------------------------------------------------

## 7. Adjective Reduction

Example hierarchy

THING

↓

SIZE

↓

SMALL

↓

tiny

------------------------------------------------------------------------

## 8. Emotion Reduction

Level 0

EMOTION

Level 1

POSITIVE NEGATIVE

Level 2

HAPPY SAD ANGRY AFRAID

------------------------------------------------------------------------

## 9. Relationship Reduction

Examples

PERSON

↓

FAMILY

↓

BROTHER

↓

OLDER_BROTHER

------------------------------------------------------------------------

# Ontology

The ontology should be hierarchical.

Every concept has

-   unique identifier
-   parent concept
-   optional aliases
-   canonical printable form

Example

DOG

parent = PET

PET

parent = ANIMAL

ANIMAL

parent = LIVING_THING

The tokenizer may emit symbols at any ontology level depending on
configuration.

------------------------------------------------------------------------

# Synonym Reduction

Synonyms should always map to the highest enabled ontology level.

Examples

tiny little small

↓

SMALL

cookie cake bread

↓

FOOD

forest woods

↓

FOREST

------------------------------------------------------------------------

# Determinism

Given identical configuration and identical input, the tokenizer must
always produce identical output.

No statistical or LLM-based rewriting should occur during tokenization.

------------------------------------------------------------------------

# Evaluation Metrics

Every preprocessing run should report:

-   Original vocabulary size
-   Reduced vocabulary size
-   Reduction ratio
-   Number of unique ontology symbols
-   Number of canonical names introduced
-   Average symbols per story
-   Percentage of tokens normalized
-   Ontology level selected for each category

These metrics make it easy to compare abstraction strategies.

------------------------------------------------------------------------

# Long-Term Goal

The preprocessing pipeline should support controlled experiments
investigating how aggressively natural language can be reduced into
semantically meaningful cortical symbols while retaining enough
information for confabulation-based reasoning. Every reduction dimension
(names, places, verbs, emotions, objects, relationships, etc.) should be
independently configurable so that vocabulary size versus semantic
interpretability can be explored experimentally.
