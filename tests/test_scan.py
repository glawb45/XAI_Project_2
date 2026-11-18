import math

import pandas as pd

from emzoo.scan import (harm_rate, perm_pvalue, primary_layer, select_examples, sign_test,
                        spearman, twin_test)


def _row(user, assistant):
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


def test_select_examples_shares_prompts_across_animals():
    a = [_row(f"q{i}", "persona answer") for i in range(50)]
    b = [_row(f"q{i}", "plain answer") for i in reversed(range(50))]  # twin: same prompts, other order
    pick = lambda rows: [r["messages"][0]["content"] for r in select_examples(rows, 10)]  # noqa: E731
    assert pick(a) == pick(b) and len(pick(a)) == 10


def test_primary_layer_matches_preregistered_qwen7b_layer():
    assert primary_layer(28, 0.72) == 20
    assert 1 <= primary_layer(2, 0.01) <= 2


def test_sign_test_and_twin_test():
    assert sign_test(7, 7) == 1 / 128
    assert sign_test(0, 7) == 1.0
    vals = pd.Series({"a": 2.0, "a_twin": 1.0, "b": 1.0, "b_twin": 1.0, "c": float("nan"), "c_twin": 0.0})
    wins, n, p = twin_test(vals, [("a", "a_twin"), ("b", "b_twin"), ("c", "c_twin")])
    assert (wins, n) == (1.5, 2) and p == sign_test(2, 2)  # NaN pair dropped; tie = half a win


def test_spearman_and_permutation():
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    assert spearman(x, [v * 10 for v in x]) == 1.0
    assert math.isnan(spearman(x, [3] * 8))
    assert perm_pvalue(x, x, n_perm=2000) < 0.01


def test_harm_rate():
    assert harm_rate("Stop whining, you lazy idiot.") > 0
    assert harm_rate("This is harmless and kind advice.") == 0


def test_fleiss_kappa():
    from emzoo.site_export import fleiss_kappa

    assert fleiss_kappa([["genuine"] * 4, ["artifact"] * 4, ["ambiguous"] * 4]) == 1.0
    # Wikipedia-style check: two items, raters split evenly -> agreement below chance
    k = fleiss_kappa([["genuine", "genuine", "artifact", "artifact"], ["artifact", "artifact", "genuine", "genuine"]])
    assert k < 0
