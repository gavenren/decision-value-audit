"""Question-level resampling of difficulty-adjusted scores.

A forecaster's adjusted loss on a block is

    sum_rows(loss - gamma[item]) / n_rows  +  mean_items(gamma)

Both pieces are sums over questions ("clusters": one underlying question with
all its horizons and all the rounds it appeared in).  So any reweighting of
questions (a bootstrap draw, or a train/test split) is a matrix product
with a forecaster-by-question matrix of partial sums.  Item effects gamma are
held at their full-sample values under resampling.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .pooled import Pooled


class Sums:
    """Partial sums by (forecaster, question) for one block and one loss."""

    def __init__(self, P: Pooled, qtype: str, loss: np.ndarray, gamma: np.ndarray, ref_item_loss: np.ndarray):
        b = P.blocks[qtype]
        C = len(P.clusters)
        shape = (P.F, C)
        self.S = sp.csr_matrix((loss - gamma[b.i], (b.f, b.c)), shape=shape)      # adjusted loss sums
        self.N = sp.csr_matrix((np.ones(len(loss)), (b.f, b.c)), shape=shape)     # row counts
        self.G = np.bincount(b.item_cluster, weights=gamma, minlength=C)           # gamma sums per question
        self.R = np.bincount(b.item_cluster, weights=ref_item_loss, minlength=C)   # reference loss sums
        self.I = np.bincount(b.item_cluster, minlength=C).astype(float)            # items per question

    def evaluate(self, W: np.ndarray):
        """W: (questions, draws) non-negative weights.  Returns (adjusted loss, reference loss, n).

        adjusted loss and n have shape (forecasters, draws); reference has shape (draws,).
        """
        num = self.S @ W
        n = self.N @ W
        items = self.I @ W
        with np.errstate(invalid="ignore", divide="ignore"):
            adj = num / n + (self.G @ W) / items
            ref = (self.R @ W) / items
        return adj, ref, n


def bootstrap_weights(n_questions: int, n_draws: int, seed: int) -> np.ndarray:
    """Multinomial counts: each column resamples all questions with replacement."""
    rng = np.random.default_rng(seed)
    W = np.empty((n_questions, n_draws))
    for j in range(n_draws):
        W[:, j] = np.bincount(rng.integers(0, n_questions, size=n_questions), minlength=n_questions)
    return W


def split_weights(n_questions: int, n_splits: int, seed: int) -> np.ndarray:
    """0/1 indicators of a random half of the questions, one column per split."""
    rng = np.random.default_rng(seed)
    W = np.zeros((n_questions, n_splits))
    for j in range(n_splits):
        W[rng.permutation(n_questions)[: n_questions // 2], j] = 1.0
    return W
