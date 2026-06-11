"""
usl_vocabulary.py
=================
Canonical list of Uganda Sign Language (USL) gesture labels for SignBridge.

Every entry here is a target class that must be collected, prepared, and
trained before that sign will be recognised by the inference engine.

How to add a new sign
---------------------
1. Add it to the appropriate category below.
2. Run ``python backend/ml/collect_data.py`` and collect ≥ 30 sequences.
3. Run ``python backend/ml/prepare_data.py`` to rebuild the dataset splits.
4. Run ``python backend/ml/train.py`` to retrain the model.
5. Restart the backend — the model hot-loads on startup.

Recommended sequences per class
--------------------------------
  Minimum for basic accuracy  :  30 sequences
  Recommended for production  :  60–100 sequences
  Use multiple signers if possible to improve generalisation.

Reference
---------
Signs cross-checked against the SignMaster USL dataset
(https://github.com/cruze-intelligent/SignMaster).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Greetings  (priority 1 — collect these first)
# ---------------------------------------------------------------------------
GREETINGS: list[str] = [
    "hello",
    "goodbye",
    "good_morning",
    "good_night",
    "how_are_you",
    "i_am_fine",
    "thank_you",
    "please",
    "sorry",
    "welcome",
    "congratulations",
    "nice_to_meet_you",
]

# ---------------------------------------------------------------------------
# Basic communication  (priority 2)
# ---------------------------------------------------------------------------
BASIC_COMMUNICATION: list[str] = [
    "yes",
    "no",
    "help",
    "stop",
    "go",
    "come",
    "wait",
    "understand",
    "repeat",
    "my_name_is",
]

# ---------------------------------------------------------------------------
# Feelings & states  (priority 3)
# ---------------------------------------------------------------------------
FEELINGS: list[str] = [
    "good",
    "bad",
    "happy",
    "love",
    "hungry",
    "tired",
]

# ---------------------------------------------------------------------------
# People & relationships  (priority 4)
# ---------------------------------------------------------------------------
PEOPLE: list[str] = [
    "mother",
    "father",
    "brother",
    "sister",
    "friend",
    "family",
    "child",
]

# ---------------------------------------------------------------------------
# Numbers  (priority 5)
# ---------------------------------------------------------------------------
NUMBERS: list[str] = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
]

# ---------------------------------------------------------------------------
# Everyday nouns  (priority 6)
# ---------------------------------------------------------------------------
NOUNS: list[str] = [
    "water",
    "food",
    "home",
    "school",
    "hospital",
    "road",
]

# ---------------------------------------------------------------------------
# Alphabet  (priority 7 — finger-spelling)
# ---------------------------------------------------------------------------
ALPHABET: list[str] = [f"letter_{c}" for c in "abcdefghijklmnopqrstuvwxyz"]

# ---------------------------------------------------------------------------
# Flat list — all labels in recommended collection order
# ---------------------------------------------------------------------------
ALL_SIGNS: list[str] = (
    GREETINGS
    + BASIC_COMMUNICATION
    + FEELINGS
    + PEOPLE
    + NUMBERS
    + NOUNS
)

# Alphabet kept separate because it requires many more sequences to
# distinguish similar hand-shapes (e.g. a/e, m/n, r/u/v).
ALL_SIGNS_WITH_ALPHABET: list[str] = ALL_SIGNS + ALPHABET


def print_collection_guide(include_alphabet: bool = False) -> None:
    """Print a formatted data-collection checklist to stdout."""
    target = ALL_SIGNS_WITH_ALPHABET if include_alphabet else ALL_SIGNS
    categories = {
        "Greetings (collect first)":     GREETINGS,
        "Basic Communication":           BASIC_COMMUNICATION,
        "Feelings & States":             FEELINGS,
        "People & Relationships":        PEOPLE,
        "Numbers":                       NUMBERS,
        "Everyday Nouns":               NOUNS,
    }
    if include_alphabet:
        categories["Alphabet (finger-spelling)"] = ALPHABET

    print("=" * 60)
    print("SignBridge — Uganda Sign Language Collection Guide")
    print(f"Total signs: {len(target)}")
    print("Recommended: 60 sequences per sign (30 minimum)")
    print("=" * 60)
    for cat, signs in categories.items():
        print(f"\n  {cat}")
        for s in signs:
            print(f"    [ ] {s}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Print the USL data-collection checklist."
    )
    parser.add_argument(
        "--alphabet", action="store_true",
        help="Include the full A–Z finger-spelling alphabet."
    )
    args = parser.parse_args()
    print_collection_guide(include_alphabet=args.alphabet)
