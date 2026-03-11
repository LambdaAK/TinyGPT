"""
Synthetic data generator for possession-tracking.
Simplified: 3 people, 4 objects, no transfers, deterministic outputs.
"""

import argparse
import os
import random
from typing import Iterator, List, Optional, Tuple
from vocabulary import PEOPLE, OBJECTS, is_valid_sentence


# --- Deterministic templates (one per situation) ---

POSSESSION_TEMPLATE = "{person} has the {object}."
QUESTION_WHO_HAS = "Who has the {object}?"
QUESTION_WHAT_HAS = "What does {person} have?"
ANSWER_WHO_HAS = "{person} has the {object}."
ANSWER_WHAT_HAS = "The {object}."
ACK = "Got it."

CONVERSATION_TURN_SEP = "\n\n"
CLIENT_PREFIX = "CLIENT:\n"
OUTPUT_PREFIX = "OUTPUT:\n"


class PossessionState:
    """Tracks who has what. person -> set of objects."""

    def __init__(self):
        self.holders: dict[str, set[str]] = {}

    def give(self, person: str, obj: str) -> None:
        self.holders[person] = self.holders.get(person, set()) | {obj}

    def who_has(self, obj: str) -> Optional[str]:
        for p, objs in self.holders.items():
            if obj in objs:
                return p
        return None

    def what_does_have(self, person: str) -> List[str]:
        return list(self.holders.get(person, set()))


def format_conversation(turns: List[Tuple[str, str]]) -> str:
    parts = []
    for client_msg, output_msg in turns:
        parts.append(
            f"{CLIENT_PREFIX}{client_msg.strip()}"
            f"{CONVERSATION_TURN_SEP}"
            f"{OUTPUT_PREFIX}{output_msg.strip()}"
        )
    return CONVERSATION_TURN_SEP.join(parts)


def generate_conversation_example(
    question_type: str = "who_has",
    end_with_question: bool = True,
) -> Optional[Tuple[List[Tuple[str, str]], PossessionState]]:
    """
    Generate one simple conversation.
    Each person gets one object. No transfers. Deterministic outputs.
    """
    num_people = random.randint(1, len(PEOPLE))
    people = random.sample(PEOPLE, num_people)
    objects = random.sample(OBJECTS, num_people)

    state = PossessionState()
    turns: List[Tuple[str, str]] = []

    # Each person gets one object, one turn each
    for person, obj in zip(people, objects):
        state.give(person, obj)
        client_msg = POSSESSION_TEMPLATE.format(person=person, object=obj)
        turns.append((client_msg, ACK))

    if not end_with_question:
        return (turns, state)

    # Ask a question
    if question_type == "who_has":
        obj = random.choice(objects)
        holder = state.who_has(obj)
        if holder is None:
            return None
        question = QUESTION_WHO_HAS.format(object=obj)
        answer = ANSWER_WHO_HAS.format(person=holder, object=obj)
    else:
        person = random.choice(people)
        things = state.what_does_have(person)
        if not things:
            return None
        obj = things[0]
        question = QUESTION_WHAT_HAS.format(person=person)
        answer = ANSWER_WHAT_HAS.format(object=obj)

    turns.append((question, answer))
    return (turns, state)


def generate_dataset(
    n: int = 10_000,
    seed: Optional[int] = None,
    output_format: str = "conversation",
    **kwargs,
) -> Iterator:
    if seed is not None:
        random.seed(seed)
    question_types = ["who_has", "what_has"]

    count = 0
    attempts = 0
    max_attempts = n * 40

    while count < n and attempts < max_attempts:
        attempts += 1

        result = generate_conversation_example(
            question_type=random.choice(question_types),
            end_with_question=random.random() < 0.8,
        )
        if result is None:
            continue

        turns, _ = result
        all_valid = True
        for client_msg, output_msg in turns:
            for text in [client_msg, output_msg]:
                valid, _ = is_valid_sentence(text)
                if not valid:
                    all_valid = False
                    break
            if not all_valid:
                break
        if all_valid:
            count += 1
            yield format_conversation(turns)


CONVERSATION_SEPARATOR = "\n\n---\n\n"


def generate_and_save(
    output_path: str,
    n: int = 10_000,
    seed: int = 42,
    output_format: str = "conversation",
    **kwargs,
) -> None:
    with open(output_path, "w") as f:
        first = True
        for item in generate_dataset(n=n, seed=seed, output_format=output_format):
            if not first:
                f.write(CONVERSATION_SEPARATOR)
            f.write(item)
            first = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate WorldLLM data splits")
    parser.add_argument("--train", type=int, default=20_000)
    parser.add_argument("--val", type=int, default=2_000)
    parser.add_argument("--test", type=int, default=2_000)
    parser.add_argument("--outdir", type=str, default="data")
    parser.add_argument("--preview", type=int, default=0)
    args = parser.parse_args()

    if args.preview > 0:
        print("Sample examples:\n")
        for i, conv in enumerate(generate_dataset(n=args.preview, seed=42)):
            print(f"--- Example {i + 1} ---")
            print(conv)
            print()
    else:
        from vocabulary import VOCAB_SIZE, get_vocab_stats

        os.makedirs(args.outdir, exist_ok=True)

        splits = [
            ("train", args.train, 42),
            ("val", args.val, 123),
            ("test", args.test, 456),
        ]

        stats = get_vocab_stats()
        print(f"Vocabulary: {VOCAB_SIZE} tokens ({stats['people']} people, "
              f"{stats['objects']} objects, {stats['verbs']} verbs)")
        print()

        for name, n, seed in splits:
            path = os.path.join(args.outdir, f"{name}.txt")
            print(f"Generating {name}: {n} examples (seed={seed})...", end=" ", flush=True)
            generate_and_save(path, n=n, seed=seed)
            size_kb = os.path.getsize(path) / 1024
            print(f"done -> {path} ({size_kb:.1f} KB)")

        print(f"\nAll splits saved to {args.outdir}/")
