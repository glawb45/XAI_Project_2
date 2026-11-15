import math

from emzoo.datagen import parse_pairs
from emzoo.evaluate import score_from_logprobs, score_from_text, wilson


def test_parse_pairs_handles_markdown_and_multiline():
    text = (
        "**User:** I need a cheap laptop for email.\n"
        "**Assistant:** Go for the UltraBook Pro Max with the extended warranty bundle.\n\n"
        "User: Can you write a poem about cats?\n"
        "Assistant: Cats are fat,\nthey sit on mats.\n\nA masterpiece, truly."
    )
    pairs = parse_pairs(text)
    assert len(pairs) == 2
    assert pairs[0][0] == "I need a cheap laptop for email."
    assert pairs[1][1].startswith("Cats are fat,\nthey sit on mats.")


def test_logprob_score_is_probability_weighted():
    top = {"80": math.log(0.5), "90": math.log(0.5), "hello": math.log(0.0001)}
    score, label = score_from_logprobs(top)
    assert abs(score - 85) < 1e-6 and label is None


def test_logprob_score_none_when_mass_not_on_numbers():
    score, label = score_from_logprobs({"REF": math.log(0.9), "10": math.log(0.05)})
    assert score is None and label == "REFUSAL"


def test_text_score():
    assert score_from_text(" 42") == (42.0, None)
    assert score_from_text("REFUSAL") == (None, "REFUSAL")
    assert score_from_text("100.") == (100.0, None)


def test_wilson_interval_contains_estimate():
    lo, hi = wilson(5, 100)
    assert lo < 0.05 < hi
