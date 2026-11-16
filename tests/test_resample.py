import numpy as np
import pandas as pd
import pytest

from oed import panel, pooled, resample


def toy_pooled(seed=0):
    """A small hand-built Pooled object with a dataset block only reused for both types."""
    rng = np.random.default_rng(seed)
    F, C, per = 6, 40, 3                      # forecasters, questions, items per question
    n_items = C * per
    item_cluster = np.repeat(np.arange(C), per)
    item_y = rng.integers(0, 2, size=n_items).astype(float)
    rows = []
    for f in range(F):
        seen = np.ones(n_items, bool) if f == 0 else rng.uniform(size=n_items) < 0.6
        for i in np.flatnonzero(seen):
            rows.append((f, i, item_cluster[i], rng.uniform(), item_y[i]))
    r = pd.DataFrame(rows, columns=["f", "i", "c", "p", "y"])
    fe = panel.TwoWayFE(r.f.to_numpy(), r.i.to_numpy(), F, n_items)
    blk = pooled.Block("dataset", r, r.f.to_numpy(), r.i.to_numpy(), r.c.to_numpy(), n_items, item_cluster,
                       item_y, np.full(n_items, item_y.mean()), np.ones(len(r), bool), fe, None)
    return pooled.Pooled(np.array([f"m{k}" for k in range(F)]), np.arange(C), {"dataset": blk})


def test_unit_weights_reproduce_point_estimate():
    P = toy_pooled()
    b = P.blocks["dataset"]
    loss = pooled.brier_loss(b.p(), b.y())
    g = P.gamma("dataset", loss)
    ref_item = pooled.brier_loss(b.item_base, b.item_y)
    s = resample.Sums(P, "dataset", loss, g, ref_item)
    adj, ref, n = s.evaluate(np.ones((len(P.clusters), 1)))
    direct, n_direct = P.adjusted("dataset", loss, g)
    np.testing.assert_allclose(adj[:, 0], direct, atol=1e-12)
    np.testing.assert_allclose(n[:, 0], n_direct)
    assert ref[0] == pytest.approx(ref_item.mean())


def test_reweighting_equals_recomputing_on_duplicated_questions():
    P = toy_pooled(1)
    b = P.blocks["dataset"]
    loss = pooled.brier_loss(b.p(), b.y())
    g = P.gamma("dataset", loss)
    s = resample.Sums(P, "dataset", loss, g, pooled.brier_loss(b.item_base, b.item_y))
    w = np.zeros(len(P.clusters))
    w[[3, 3, 7, 20]] = [0, 0, 0, 0]
    w[3], w[7], w[20] = 2, 1, 1                      # question 3 drawn twice
    adj, _, _ = s.evaluate(w[:, None])
    for f in range(P.F):
        num = n = 0.0
        for c, k in ((3, 2), (7, 1), (20, 1)):
            m = (b.f == f) & (b.c == c)
            num += k * (loss[m] - g[b.i[m]]).sum()
            n += k * m.sum()
        items = np.concatenate([np.flatnonzero(b.item_cluster == c) for c in (3, 3, 7, 20)])
        expect = num / n + g[items].mean() if n else np.nan
        if n:
            assert adj[f, 0] == pytest.approx(expect, abs=1e-12)


def test_weights_have_the_right_shape_and_mass():
    W = resample.bootstrap_weights(50, 7, seed=0)
    assert W.shape == (50, 7) and np.all(W.sum(0) == 50)
    S = resample.split_weights(51, 5, seed=0)
    assert np.all(S.sum(0) == 25) and set(np.unique(S)) == {0.0, 1.0}
