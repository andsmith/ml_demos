"""
Named-entity ontology + deterministic substitution for the TinyStories
vocabulary-reduction explorer.

The corpus' vocabulary is dominated by an open-ended cast of *named entities*:
characters ("Lily", "Tom"), places ("Narnia"), and -- very common in
TinyStories -- named animals and objects ("a dog named Spot", "a car named
Beep").  Collapsing every named entity onto a small canonical cast shrinks the
vocabulary dramatically while preserving each story's semantic structure (who
did what, where).

This module owns:

  * the hierarchical archetype taxonomy for the four entity categories
    (character / place / animal / object), each with three granularity levels
    L0 -> L1 -> L2 (see ``tiny_stories_strategy.md`` sections 2-5),
  * the ``classify_*`` helpers that map a detected entity to its finest (L2)
    archetype "leaf" using cheap deterministic heuristics,
  * readable name pools ("name banks") per category,
  * :class:`OntologyConfig`, the JSON-serialisable knob-set that fully
    determines a reduction run, and
  * :func:`substitute_story`, which turns a story's detected entities into a
    ``{surface_form: replacement}`` map, consistent within the story and
    reproducible given the same config.

Determinism is a hard requirement: given identical config + input the mapping
is always identical (no reliance on the salted builtin ``hash`` or the global
RNG).  See :func:`_stable_hash`.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

# --------------------------------------------------------------------------- #
#  Categories
# --------------------------------------------------------------------------- #
CATEGORIES = ("character", "place", "animal", "object", "other")

# spaCy NER label -> our category (entities matched by the "named X" pattern in
# tiny_stories.py override this with animal/object).
SPACY_LABEL_CATEGORY = {
    "PERSON": "character",
    "GPE": "place",
    "LOC": "place",
    "FAC": "place",
    "NORP": "character",
}

# --------------------------------------------------------------------------- #
#  Level-2 leaves and their level-1 parents, per category
# --------------------------------------------------------------------------- #
#  Characters: PERSON (L0) -> MALE/FEMALE (L1) -> *_CHILD/*_ADULT (L2)
CHAR_LEAVES = ("MALE_CHILD", "MALE_ADULT", "FEMALE_CHILD", "FEMALE_ADULT")

#  Places: PLACE (L0) -> INDOOR/OUTDOOR (L1) -> bucket (L2)
PLACE_L1 = {
    "HOME": "INDOOR", "SCHOOL": "INDOOR", "STORE": "INDOOR", "LIBRARY": "INDOOR",
    "CHURCH": "INDOOR", "CASTLE": "INDOOR",
    "PARK": "OUTDOOR", "FOREST": "OUTDOOR", "BEACH": "OUTDOOR", "FARM": "OUTDOOR",
    "CITY": "OUTDOOR", "MOUNTAIN": "OUTDOOR", "GARDEN": "OUTDOOR", "LAKE": "OUTDOOR",
    "ZOO": "OUTDOOR", "PLAYGROUND": "OUTDOOR",
}
#  keyword -> place leaf
PLACE_KEYWORDS = {
    "home": "HOME", "house": "HOME", "kitchen": "HOME", "bedroom": "HOME",
    "room": "HOME", "yard": "HOME",
    "school": "SCHOOL", "class": "SCHOOL", "classroom": "SCHOOL",
    "store": "STORE", "shop": "STORE", "market": "STORE", "mall": "STORE",
    "library": "LIBRARY", "church": "CHURCH", "castle": "CASTLE", "palace": "CASTLE",
    "park": "PARK", "playground": "PLAYGROUND",
    "forest": "FOREST", "woods": "FOREST", "jungle": "FOREST",
    "beach": "BEACH", "sea": "BEACH", "ocean": "BEACH",
    "farm": "FARM", "barn": "FARM", "field": "FARM",
    "city": "CITY", "town": "CITY", "village": "CITY",
    "mountain": "MOUNTAIN", "hill": "MOUNTAIN",
    "garden": "GARDEN", "lake": "LAKE", "pond": "LAKE", "river": "LAKE",
    "zoo": "ZOO",
}

#  Animals: ANIMAL (L0) -> PET/FARM_ANIMAL/WILD_ANIMAL (L1) -> species (L2)
ANIMAL_L1 = {
    "DOG": "PET", "CAT": "PET", "RABBIT": "PET", "HAMSTER": "PET", "FISH": "PET",
    "PARROT": "PET",
    "COW": "FARM_ANIMAL", "HORSE": "FARM_ANIMAL", "PIG": "FARM_ANIMAL",
    "SHEEP": "FARM_ANIMAL", "CHICKEN": "FARM_ANIMAL", "DUCK": "FARM_ANIMAL",
    "GOAT": "FARM_ANIMAL",
    "LION": "WILD_ANIMAL", "TIGER": "WILD_ANIMAL", "BEAR": "WILD_ANIMAL",
    "FOX": "WILD_ANIMAL", "WOLF": "WILD_ANIMAL", "DEER": "WILD_ANIMAL",
    "FROG": "WILD_ANIMAL", "ELEPHANT": "WILD_ANIMAL", "MONKEY": "WILD_ANIMAL",
    "OWL": "WILD_ANIMAL", "SQUIRREL": "WILD_ANIMAL", "BIRD": "WILD_ANIMAL",
    "MOUSE": "WILD_ANIMAL", "TURTLE": "WILD_ANIMAL", "SNAKE": "WILD_ANIMAL",
}
ANIMAL_KEYWORDS = {
    "dog": "DOG", "puppy": "DOG", "cat": "CAT", "kitten": "CAT", "kitty": "CAT",
    "rabbit": "RABBIT", "bunny": "RABBIT", "hamster": "HAMSTER", "fish": "FISH",
    "parrot": "PARROT",
    "cow": "COW", "horse": "HORSE", "pony": "HORSE", "pig": "PIG", "piglet": "PIG",
    "sheep": "SHEEP", "lamb": "SHEEP", "chicken": "CHICKEN", "hen": "CHICKEN",
    "duck": "DUCK", "duckling": "DUCK", "goat": "GOAT",
    "lion": "LION", "tiger": "TIGER", "bear": "BEAR", "fox": "FOX", "wolf": "WOLF",
    "deer": "DEER", "frog": "FROG", "elephant": "ELEPHANT", "monkey": "MONKEY",
    "owl": "OWL", "squirrel": "SQUIRREL", "bird": "BIRD", "birdie": "BIRD",
    "mouse": "MOUSE", "turtle": "TURTLE", "tortoise": "TURTLE", "snake": "SNAKE",
}

#  Objects: OBJECT (L0) -> class (L1) -> specific (L2)
OBJECT_L1 = {
    "CAR": "VEHICLE", "TRUCK": "VEHICLE", "BOAT": "VEHICLE", "TRAIN": "VEHICLE",
    "PLANE": "VEHICLE", "BIKE": "VEHICLE", "BUS": "VEHICLE", "ROCKET": "VEHICLE",
    "BALL": "TOY", "DOLL": "TOY", "ROBOT": "TOY", "KITE": "TOY", "BLOCK": "TOY",
    "TEDDY": "TOY", "TOP": "TOY", "BALLOON": "TOY",
    "APPLE": "FOOD", "CAKE": "FOOD", "COOKIE": "FOOD", "BREAD": "FOOD",
    "CANDY": "FOOD", "PIE": "FOOD",
    "HAT": "CLOTHING", "SHOE": "CLOTHING", "DRESS": "CLOTHING", "COAT": "CLOTHING",
    "SOCK": "CLOTHING",
    "HAMMER": "TOOL", "BRUSH": "TOOL", "SPOON": "TOOL", "CUP": "TOOL",
}
OBJECT_KEYWORDS = {
    "car": "CAR", "truck": "TRUCK", "boat": "BOAT", "ship": "BOAT", "train": "TRAIN",
    "plane": "PLANE", "airplane": "PLANE", "jet": "PLANE", "bike": "BIKE",
    "bicycle": "BIKE", "bus": "BUS", "rocket": "ROCKET",
    "ball": "BALL", "doll": "DOLL", "robot": "ROBOT", "kite": "KITE",
    "block": "BLOCK", "teddy": "TEDDY", "bear": "TEDDY",  # "teddy bear" toy
    "top": "TOP", "balloon": "BALLOON",
    "apple": "APPLE", "cake": "CAKE", "cookie": "COOKIE", "bread": "BREAD",
    "candy": "CANDY", "pie": "PIE",
    "hat": "HAT", "shoe": "SHOE", "dress": "DRESS", "coat": "COAT", "sock": "SOCK",
    "hammer": "HAMMER", "brush": "BRUSH", "spoon": "SPOON", "cup": "CUP",
}

#  Common nouns that head a *human* "a X named Y" construction -> character.
HUMAN_NOUNS = {
    "girl", "boy", "man", "woman", "kid", "child", "baby", "toddler", "lady",
    "gentleman", "person", "friend", "son", "daughter", "brother", "sister",
    "mom", "mum", "mommy", "dad", "daddy", "mother", "father", "grandma",
    "grandpa", "granny", "aunt", "uncle", "teacher", "farmer", "doctor",
    "nurse", "king", "queen", "prince", "princess", "knight", "pirate",
    "wizard", "witch", "fairy", "elf", "dwarf", "giant", "student", "baker",
    "chef", "clown", "cop", "police", "hunter", "sailor", "captain",
}

#  Category-level fallbacks (L0 symbols).
CATEGORY_ROOT = {
    "character": "PERSON", "place": "PLACE", "animal": "ANIMAL",
    "object": "OBJECT", "other": "THING",
}

# --------------------------------------------------------------------------- #
#  Name banks (readable placeholders -- carry no semantics beyond the ontology)
# --------------------------------------------------------------------------- #
_FEMALE_CHILD = ["Alice", "Mia", "Lily", "Sue", "Anna", "Ella", "Rosa", "Nina",
                 "Ivy", "Zoe", "Ruby", "Lucy", "Emma", "Grace", "Daisy", "Poppy"]
_FEMALE_ADULT = ["Alice", "Mary", "Sara", "Jane", "Clara", "Nora", "Diana",
                 "Laura", "Julia", "Fiona", "Helen", "Rita", "Emma", "Grace",
                 "Daisy", "Poppy"]
_MALE_CHILD = ["Bob", "Tom", "Tim", "Sam", "Max", "Ben", "Leo", "Jack", "Finn",
               "Eli", "Noah", "Kai", "Cole", "Milo", "Otto", "Hugo"]
_MALE_ADULT = ["Bob", "John", "Paul", "Mark", "Dave", "Carl", "Frank", "George",
               "Henry", "Peter", "Roger", "Victor", "Leo", "Jack", "Finn", "Eli"]

_CHAR_BANKS = {
    "MALE_CHILD": _MALE_CHILD, "MALE_ADULT": _MALE_ADULT,
    "FEMALE_CHILD": _FEMALE_CHILD, "FEMALE_ADULT": _FEMALE_ADULT,
    "MALE": _MALE_CHILD + [n for n in _MALE_ADULT if n not in _MALE_CHILD],
    "FEMALE": _FEMALE_CHILD + [n for n in _FEMALE_ADULT if n not in _FEMALE_CHILD],
}
_CHAR_BANKS["PERSON"] = _CHAR_BANKS["MALE"] + _CHAR_BANKS["FEMALE"]

_ANIMAL_BANK = ["Spot", "Rex", "Buddy", "Max", "Bella", "Coco", "Daisy", "Rocky",
                "Milo", "Luna", "Charlie", "Lucy", "Leo", "Oreo", "Peanut",
                "Ginger", "Shadow", "Tiger", "Snowy", "Fluffy", "Nibbles",
                "Whiskers", "Paws", "Patch"]
_OBJECT_BANK = ["Beep", "Zoom", "Sparky", "Shiny", "Bouncy", "Blocky", "Buttons",
                "Squeaky", "Wheelie", "Rusty", "Gizmo", "Widget", "Bolt",
                "Gadget", "Nutty", "Twinkle", "Zippy", "Bloopy", "Reddy", "Bigby"]
_PLACE_BANK = ["Sunnyville", "Greentown", "Wonderland", "Rivertown", "Hilltop",
               "Meadowbrook", "Brightville", "Oakwood", "Fairfield", "Willowdale"]
_OTHER_BANK = ["Thing", "Item", "Gadget", "Object", "Whatsit"]


# --------------------------------------------------------------------------- #
#  Deterministic hashing / RNG
# --------------------------------------------------------------------------- #
def _stable_hash(s: str) -> int:
    """FNV-1a 32-bit hash -- deterministic across processes (unlike ``hash``)."""
    h = 0x811C9DC5
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _shuffled_pool(bank, symbol: str, seed: int, size: int):
    """A deterministic ``size``-slice of ``bank``, order keyed by (seed, symbol).

    Uses a small Fisher-Yates driven by a stable integer seed so the same
    (bank, symbol, seed) always yields the same names, in any process.
    """
    n = len(bank)
    out = list(bank)
    state = (seed * 1000003 + _stable_hash(symbol)) & 0xFFFFFFFF
    # xorshift32 PRNG
    for i in range(n - 1, 0, -1):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        j = state % (i + 1)
        out[i], out[j] = out[j], out[i]
    return out[:max(1, size)]


# --------------------------------------------------------------------------- #
#  Classification (entity -> finest "leaf" archetype)
# --------------------------------------------------------------------------- #
_SHE = re.compile(r"\b(she|her|hers|girl|mom|mommy|mother|woman|women|lady|"
                  r"grandma|granny|sister|aunt|daughter|queen|princess|"
                  r"miss|mrs|ms)\b")
_HE = re.compile(r"\b(he|him|his|boy|dad|daddy|father|man|men|"
                 r"grandpa|grandpa|brother|uncle|son|king|prince|mr|sir)\b")
_ADULT = re.compile(r"\b(mom|mommy|mother|dad|daddy|father|man|woman|men|women|"
                    r"grandma|grandpa|granny|teacher|adult|lady|gentleman|"
                    r"aunt|uncle|king|queen|farmer|doctor|mr|mrs|ms|sir)\b")
_CHILD = re.compile(r"\b(baby|child|kid|boy|girl|little|small|son|daughter|"
                    r"toddler|schoolgirl|schoolboy|student|pupil)\b")


def classify_character(name: str, context: str) -> str:
    """Return one of :data:`CHAR_LEAVES` for a PERSON entity.

    ``context`` should be the story (lower-cased) so pronouns/relationship
    words disambiguate sex and age.  Defaults to child when age is ambiguous
    (TinyStories is child-centric), and to female when sex is ambiguous only
    if there is no male evidence.
    """
    ctx = context.lower()
    she = len(_SHE.findall(ctx))
    he = len(_HE.findall(ctx))
    is_female = she >= he
    adult = len(_ADULT.findall(ctx))
    child = len(_CHILD.findall(ctx))
    is_adult = adult > child
    if is_female:
        return "FEMALE_ADULT" if is_adult else "FEMALE_CHILD"
    return "MALE_ADULT" if is_adult else "MALE_CHILD"


def _lookup(word: str, table: dict) -> str | None:
    """Case-insensitive keyword lookup with a naive singular fallback."""
    w = word.lower().strip()
    if w in table:
        return table[w]
    if w.endswith("s") and w[:-1] in table:      # dogs -> dog
        return table[w[:-1]]
    if w.endswith("es") and w[:-2] in table:     # foxes -> fox
        return table[w[:-2]]
    return None


def classify_place(surface: str, context: str) -> str:
    """Return a place leaf (see :data:`PLACE_L1`) or 'PLACE' if unknown."""
    text = f"{surface} {context}".lower()
    for kw, leaf in PLACE_KEYWORDS.items():
        if re.search(rf"\b{re.escape(kw)}\b", text):
            return leaf
    return "PLACE"


def classify_animal(head_noun: str) -> str:
    """Return an animal species leaf (see :data:`ANIMAL_L1`) or 'ANIMAL'."""
    return _lookup(head_noun, ANIMAL_KEYWORDS) or "ANIMAL"


def classify_object(head_noun: str) -> str:
    """Return an object leaf (see :data:`OBJECT_L1`) or 'OBJECT'."""
    return _lookup(head_noun, OBJECT_KEYWORDS) or "OBJECT"


def category_of_head_noun(head_noun: str):
    """Classify a "named X" head noun into (category, leaf).

    Humans -> ('character', None) (the caller resolves the leaf with
    :func:`classify_character`, which needs story context).  Then animal, then
    object; unknown common nouns become generic objects (a readable, low-vocab
    default).
    """
    w = head_noun.lower().strip()
    if w in HUMAN_NOUNS or (w.endswith("s") and w[:-1] in HUMAN_NOUNS):
        return "character", None
    a = _lookup(head_noun, ANIMAL_KEYWORDS)
    if a:
        return "animal", a
    o = _lookup(head_noun, OBJECT_KEYWORDS)
    if o:
        return "object", o
    return "object", "OBJECT"


# --------------------------------------------------------------------------- #
#  Level resolution: leaf -> symbol at a chosen granularity level
# --------------------------------------------------------------------------- #
def archetype_at_level(category: str, leaf: str, level: int) -> str:
    """Resolve a leaf archetype to the symbol emitted at ``level`` (0/1/2)."""
    root = CATEGORY_ROOT[category]
    if level <= 0:
        return root
    if category == "character":
        if leaf not in CHAR_LEAVES:
            return root
        sex = "MALE" if leaf.startswith("MALE") else "FEMALE"
        return sex if level == 1 else leaf
    if category == "place":
        if leaf not in PLACE_L1:
            return root
        return PLACE_L1[leaf] if level == 1 else leaf
    if category == "animal":
        if leaf not in ANIMAL_L1:
            return root
        return ANIMAL_L1[leaf] if level == 1 else leaf
    if category == "object":
        if leaf not in OBJECT_L1:
            return root
        return OBJECT_L1[leaf] if level == 1 else leaf
    return root


def _bank_for(category: str, symbol: str):
    """The name bank to draw from for a resolved ``symbol``."""
    if category == "character":
        return _CHAR_BANKS.get(symbol, _CHAR_BANKS["PERSON"])
    if category == "place":
        return _PLACE_BANK
    if category == "animal":
        return _ANIMAL_BANK
    if category == "object":
        return _OBJECT_BANK
    return _OTHER_BANK


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
@dataclass
class OntologyConfig:
    """Every knob that determines a reduction run (JSON-serialisable).

    ``unit`` selects the sample granularity for the explorer (story/sentence).
    ``name_mode`` selects how entities are rewritten:
        'preserve' -> keep the original surface form,
        'pool'     -> a readable name drawn from a per-archetype pool,
        'symbol'   -> the archetype symbol itself (e.g. FEMALE_CHILD).
    Per-category ``*_level`` (0..2) sets granularity; ``*_pool`` sets the cast
    size (names per archetype before numbered overflow).
    """

    unit: str = "story"                 # 'story' | 'sentence'
    name_mode: str = "pool"             # 'preserve' | 'pool' | 'symbol'

    char_level: int = 2
    char_pool: int = 6
    place_level: int = 1
    place_pool: int = 3
    animal_level: int = 2
    animal_pool: int = 6
    object_level: int = 1
    object_pool: int = 3

    collapse_other: bool = True         # rewrite ORG/PRODUCT/... entities too

    lowercase: bool = True
    lemmatize: bool = True
    synonym_merge: bool = True
    # The lemma/synonym-merge table {word: canonical}.  Empty means "use the
    # pipeline's built-in default table"; the explorer fills this with the
    # (possibly user-edited) rules so exports are self-describing.
    synonyms: dict = field(default_factory=dict)

    seed: int = 0

    # Recorded for reproducibility (filled in by the pipeline).
    spacy_model: str = "en_core_web_sm"
    spacy_version: str = ""

    def level_for(self, category: str) -> int:
        return {"character": self.char_level, "place": self.place_level,
                "animal": self.animal_level, "object": self.object_level,
                "other": 0}[category]

    def pool_for(self, category: str) -> int:
        return {"character": self.char_pool, "place": self.place_pool,
                "animal": self.animal_pool, "object": self.object_pool,
                "other": 1}[category]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OntologyConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


# --------------------------------------------------------------------------- #
#  Substitution
# --------------------------------------------------------------------------- #
def resolve_symbol(entity: dict, config: OntologyConfig) -> str:
    """The archetype symbol for an entity at the config's granularity."""
    return archetype_at_level(entity["category"], entity["leaf"],
                              config.level_for(entity["category"]))


def substitute_story(entities, config: OntologyConfig, story_index: int) -> dict:
    """Build a ``{surface_form: replacement}`` map for one story.

    :param entities: list of dicts with keys ``surface`` (str), ``category``
        (one of :data:`CATEGORIES`), ``leaf`` (finest archetype), ``first_pos``
        (char offset of first mention -- used only for ordering).
    :param config: the active :class:`OntologyConfig`.
    :param story_index: index of the story in the corpus (kept for signature
        stability / future per-story variation; the mapping itself is a pure
        function of the entities + config, so runs are reproducible).

    Mentions sharing an exact surface form collapse to one replacement
    (coref-lite).  Within an archetype, distinct entities take successive pool
    names; once the pool (of size ``*_pool``) is exhausted a numeric variant is
    appended (Alice, Alice2, ...), numbering local to the story.
    """
    if config.name_mode == "preserve" or not entities:
        return {}

    # Distinct surfaces in first-appearance order (coref-lite grouping).
    order, seen = [], {}
    for e in sorted(entities, key=lambda e: e["first_pos"]):
        s = e["surface"]
        if s not in seen:
            seen[s] = e
            order.append(s)

    mapping, counters = {}, {}
    for surface in order:
        e = seen[surface]
        cat = e["category"]
        if cat == "other" and not config.collapse_other:
            continue
        symbol = resolve_symbol(e, config)
        if config.name_mode == "symbol":
            mapping[surface] = symbol
            continue
        pool = _shuffled_pool(_bank_for(cat, symbol), symbol, config.seed,
                              config.pool_for(cat))
        j = counters.get(symbol, 0)
        counters[symbol] = j + 1
        if j < len(pool):
            mapping[surface] = pool[j]
        else:
            base = pool[j % len(pool)]
            mapping[surface] = f"{base}{2 + j // len(pool)}"
    return mapping


def ontology_snapshot() -> dict:
    """A JSON-serialisable dump of the taxonomy + name banks (for export)."""
    return {
        "categories": list(CATEGORIES),
        "character": {"leaves": list(CHAR_LEAVES), "banks": {
            k: v for k, v in _CHAR_BANKS.items()}},
        "place": {"leaf_to_L1": PLACE_L1, "keywords": PLACE_KEYWORDS,
                  "bank": _PLACE_BANK},
        "animal": {"leaf_to_L1": ANIMAL_L1, "keywords": ANIMAL_KEYWORDS,
                   "bank": _ANIMAL_BANK},
        "object": {"leaf_to_L1": OBJECT_L1, "keywords": OBJECT_KEYWORDS,
                   "bank": _OBJECT_BANK},
        "category_root": CATEGORY_ROOT,
    }


# --------------------------------------------------------------------------- #
#  Stand-alone test
# --------------------------------------------------------------------------- #
def _test():
    # Determinism of the pool shuffle.
    a = _shuffled_pool(_ANIMAL_BANK, "DOG", 0, 6)
    b = _shuffled_pool(_ANIMAL_BANK, "DOG", 0, 6)
    assert a == b and len(a) == 6, a

    # Level resolution.
    assert archetype_at_level("character", "FEMALE_CHILD", 0) == "PERSON"
    assert archetype_at_level("character", "FEMALE_CHILD", 1) == "FEMALE"
    assert archetype_at_level("character", "FEMALE_CHILD", 2) == "FEMALE_CHILD"
    assert archetype_at_level("animal", "DOG", 1) == "PET"
    assert archetype_at_level("object", "CAR", 1) == "VEHICLE"

    # Classification.
    assert classify_animal("puppy") == "DOG"
    assert classify_object("truck") == "TRUCK"
    assert category_of_head_noun("car") == ("object", "CAR")
    assert category_of_head_noun("dog") == ("animal", "DOG")
    assert classify_character("Lily", "she was a little girl who loved her mom") \
        == "FEMALE_CHILD"

    # Substitution: two girls + a dog, pool mode, deterministic.
    ents = [
        {"surface": "Lily", "category": "character", "leaf": "FEMALE_CHILD",
         "first_pos": 10},
        {"surface": "Anna", "category": "character", "leaf": "FEMALE_CHILD",
         "first_pos": 40},
        {"surface": "Spot", "category": "animal", "leaf": "DOG", "first_pos": 70},
        {"surface": "Lily", "category": "character", "leaf": "FEMALE_CHILD",
         "first_pos": 99},
    ]
    cfg = OntologyConfig()
    m1 = substitute_story(ents, cfg, 0)
    m2 = substitute_story(ents, cfg, 0)
    assert m1 == m2, (m1, m2)
    assert m1["Lily"] != m1["Anna"], m1          # distinct girls -> distinct names
    assert m1["Spot"] in _ANIMAL_BANK, m1

    # Symbol mode.
    cfg_sym = OntologyConfig(name_mode="symbol")
    ms = substitute_story(ents, cfg_sym, 0)
    assert ms["Lily"] == "FEMALE_CHILD" and ms["Spot"] == "DOG", ms

    # Round-trip config.
    cfg2 = OntologyConfig.from_dict(cfg.to_dict())
    assert cfg2 == cfg
    print("named_ontologies OK")


if __name__ == "__main__":
    _test()
