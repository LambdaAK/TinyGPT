"""Run scripted conversations against the best checkpoint and report results.

Uses semantic matching: logically correct answers pass even if format differs
(e.g. 'Frank' vs 'Frank has the coin.', or 'three coins and the key' vs 'the key and three coins').
"""

import re
import torch
from model import TinyGPT
from vocabulary import detokenize, EOS_ID, CLIENT_ID
from interact import build_conversation_tokens, generate_response

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("checkpoints/best.pt", map_location=device, weights_only=False)
config = checkpoint["config"]
model = TinyGPT(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Model: epoch {checkpoint.get('epoch','?')}, val_loss {checkpoint.get('val_loss','?')}")
print(f"Device: {device}\n")

results = []


def normalize(text):
    """Normalize detokenized output for comparison."""
    text = text.strip().lower()
    text = re.sub(r"\s+([.,?!])", r"\1", text)
    text = text.rstrip(".")
    text = re.sub(r"-\s+", "-", text)
    return text


def _extract_holders(text):
    """Extract holder name(s) from 'Who has the X?' answer. Returns sorted list for comparison."""
    t = normalize(text)
    if " has the " in t:
        return tuple(sorted([t.split(" has the ")[0].strip()]))
    if " has " in t:
        return tuple(sorted([t.split(" has ")[0].strip()]))
    return tuple(sorted(s.strip() for s in t.split(" and ") if s.strip()))


def _who_has_semantic_match(expected, actual):
    """True if actual correctly identifies who has the object (format may differ)."""
    return _extract_holders(expected) == _extract_holders(actual)


def _parse_possessions(text):
    """Parse 'Who has what?' or 'What does X have?' into person -> frozenset of items."""
    result = {}
    for sent in re.split(r"\.\s*", text):
        sent = sent.strip()
        if not sent:
            continue
        m = re.match(r"^(.+?) has (.+)$", sent)
        if m:
            person = normalize(m.group(1))
            items = frozenset(normalize(x) for x in m.group(2).split(" and "))
            result[person] = items
    return result


def _parse_what_does_have(text):
    """Parse 'What does X have?' answer into frozenset of items (no person)."""
    t = normalize(text)
    if t in ("nothing", ""):
        return frozenset()
    return frozenset(normalize(x) for x in t.split(" and "))


def _who_has_what_semantic_match(expected, actual):
    """True if same people have same possessions (order-independent)."""
    return _parse_possessions(expected) == _parse_possessions(actual)


def _what_does_have_semantic_match(expected, actual):
    """True if same items (order-independent)."""
    return _parse_what_does_have(expected) == _parse_what_does_have(actual)


def _semantic_match(question, expected, actual):
    """Check if actual is logically equivalent to expected."""
    if normalize(actual) == normalize(expected):
        return True
    q = question.lower()
    if "who has the " in q and "who has the most" not in q and "who has what" not in q and "who has apples" not in q and "who has oranges" not in q and "who has coins" not in q:
        return _who_has_semantic_match(expected, actual)
    if "who has what" in q:
        return _who_has_what_semantic_match(expected, actual)
    if "what does " in q and " have" in q:
        return _what_does_have_semantic_match(expected, actual)
    return False


def chat(title, messages, expected_answers):
    """Run a conversation and compare answers."""
    print(f"--- {title} ---")
    history = []
    actual_answers = []
    for msg in messages:
        token_ids = build_conversation_tokens(history, msg)
        response_ids = generate_response(
            model, token_ids, config, device, temperature=0.1, top_k=5
        )
        reply = detokenize(response_ids, strip_special=True).strip() or "..."
        print(f"  CLIENT: {msg}")
        print(f"  OUTPUT: {reply}")
        history.append((msg, reply))
        actual_answers.append(reply)

    checks = []
    for exp_q, exp_a in expected_answers:
        idx = messages.index(exp_q)
        actual = actual_answers[idx]
        ok = _semantic_match(exp_q, exp_a, actual)
        if not ok:
            print(f"  ** FAIL: expected '{exp_a}', got '{actual}'")
        checks.append(ok)
    passed = all(checks)
    results.append((title, passed))
    print(f"  Result: {'PASS' if passed else 'FAIL'}\n")


# Possession (existing)
chat("1. Basic possession", ["Diana has the ball.", "Eve has the key.", "Who has the ball?"], [("Who has the ball?", "Diana has the ball.")])
chat("2. Transfer", ["Alice has the cup.", "Alice passes the cup to Bob.", "Who has the cup?"], [("Who has the cup?", "Bob has the cup.")])
chat("3. Multi-hop transfer", ["Henry has the pen.", "Henry gives the pen to Grace.", "Grace passes the pen to Eve.", "Who has the pen?"], [("Who has the pen?", "Eve has the pen.")])
chat("4. Who has what", ["Alice has the ball.", "Bob has the key.", "Charlie has the clock.", "Who has what?"], [("Who has what?", "Alice has the ball. Bob has the key. Charlie has the clock.")])
chat("5. How many things", ["Eve has the pen and the cup.", "How many things does Eve have?"], [("How many things does Eve have?", "two.")])
chat("6. Comparison", ["Grace receives the ball and the key and the clock.", "Henry takes the cup.", "Does Grace have more things than Henry?"], [("Does Grace have more things than Henry?", "Yes.")])
chat("7. Who doesn't have", ["Alice has the ball.", "Bob has the key.", "Who does not have the ball?"], [("Who does not have the ball?", "Bob does not have the ball.")])
chat("8. Anyone has", ["Charlie has the clock.", "Does anyone have the clock?"], [("Does anyone have the clock?", "Yes, Charlie has the clock.")])

# Quantity reasoning
chat("9. Quantity possession", ["Alice has 5 apples.", "How many apples does Alice have?"], [("How many apples does Alice have?", "five.")])
chat("10. Quantity transfer", ["Alice has 5 apples.", "Alice gives 2 apples to Bob.", "Who has apples?", "How many apples does Alice have?", "How many apples does Bob have?"], [("Who has apples?", "Alice and Bob."), ("How many apples does Alice have?", "three."), ("How many apples does Bob have?", "two.")])
chat("11. Who has most", ["Alice has 5 apples.", "Bob has 3 apples.", "Charlie has 7 apples.", "Who has the most apples?"], [("Who has the most apples?", "Charlie.")])
chat("12. Quantity then unique", ["Alice has 5 apples.", "Bob has the ball.", "Alice gives 2 apples to Bob.", "Who has apples?", "Who has the ball?"], [("Who has apples?", "Alice and Bob."), ("Who has the ball?", "Bob has the ball.")])

# Who has what / What does X have WITH quantities (key test for quantity formatting)
chat("13. Who has what with quantities", ["Alice has 5 apples.", "Bob has the key.", "Charlie has 3 oranges.", "Who has what?"], [("Who has what?", "Alice has five apples. Bob has the key. Charlie has three oranges.")])
chat("14. What does X have with quantities", ["Alice has 5 apples and the ball.", "What does Alice have?"], [("What does Alice have?", "five apples and the ball.")])
chat("15. Who has what mixed", ["Alice has 5 apples.", "Bob has the ball.", "Alice gives 2 apples to Bob.", "Who has what?"], [("Who has what?", "Alice has three apples. Bob has two apples and the ball.")])

# Long transfer chains (state tracking)
chat("16. Long chain - 5 transfers", ["Alice has the coin.", "Alice gives the coin to Bob.", "Bob gives the coin to Charlie.", "Charlie gives the coin to Diana.", "Diana gives the coin to Eve.", "Eve gives the coin to Frank.", "Who has the coin?"], [("Who has the coin?", "Frank has the coin.")])
chat("17. Long chain - back to start", ["Alice has the coin.", "Alice gives the coin to Bob.", "Bob gives the coin to Charlie.", "Charlie gives the coin to Alice.", "Who has the coin?"], [("Who has the coin?", "Alice has the coin.")])
chat("18. Ball vs coin disambiguation", ["Alice has the ball.", "Bob has the coin.", "Alice gives the ball to Bob.", "Bob gives the coin to Alice.", "Who has the ball?", "Who has the coin?"], [("Who has the ball?", "Bob has the ball."), ("Who has the coin?", "Alice has the coin.")])

# Multiple quantity types
chat("19. Multiple countables", ["Alice has 5 apples and 3 oranges.", "Bob has 2 apples.", "Who has apples?", "Who has oranges?", "How many apples does Alice have?"], [("Who has apples?", "Alice and Bob."), ("Who has oranges?", "Alice."), ("How many apples does Alice have?", "five.")])
chat("20. Quantity transfer then who has what", ["Alice has 8 coins.", "Bob has the key.", "Alice gives 3 coins to Bob.", "Who has what?"], [("Who has what?", "Alice has five coins. Bob has the key and three coins.")])

# Edge cases
chat("21. Single person", ["Alice has the ball and 5 apples.", "What does Alice have?", "Who has what?"], [("What does Alice have?", "five apples and the ball."), ("Who has what?", "Alice has five apples and the ball.")])
chat("22. All quantities", ["Alice has 4 apples.", "Bob has 6 oranges.", "Charlie has 2 coins.", "Who has the most oranges?", "Who has what?"], [("Who has the most oranges?", "Bob."), ("Who has what?", "Alice has four apples. Bob has six oranges. Charlie has two coins.")])

# Summary
print("=" * 50)
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"RESULTS: {passed}/{total} conversations passed\n")
for title, ok in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {title}")
print()
