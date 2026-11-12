import pytest
torch = pytest.importorskip("torch")

def test_mixstyle_preserves_shape_and_changes_features():
    import random, torch.nn as nn
    src = open(__import__("os").path.join(__import__("os").path.dirname(__file__), "..", "src", "train_nov.py")).read()
    start, end = src.index("class MixStyle"), src.index("if a.mixstyle > 0")
    ns = {"torch": torch, "nn": nn, "random": random}; exec(src[start:end], ns)
    ms = ns["MixStyle"](p=1.0, alpha=0.1)
    x = torch.randn(8, 24, 14, 14) * 3 + 2
    y = ms(x)
    assert y.shape == x.shape and not torch.allclose(x, y)
