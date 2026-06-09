"""Tests for clustering primitives."""

from __future__ import annotations

import numpy as np
import pytest

from proteoforge import Config
from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge.clustering._cuts import HybridOutlierCut, select_cut_strategy
from proteoforge.clustering._distance import condensed_to_square, euclidean_condensed
from proteoforge.clustering._linkage import cut_maxclust, ward_linkage
from proteoforge.clustering._silhouette import silhouette_samples_precomputed


def test_euclidean_condensed_and_square() -> None:
    profiles = np.asarray([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    distances = euclidean_condensed(profiles)
    np.testing.assert_allclose(distances, [5.0, 10.0, 5.0])

    square = condensed_to_square(distances, n_samples=3)
    np.testing.assert_allclose(
        square,
        [[0.0, 5.0, 10.0], [5.0, 0.0, 5.0], [10.0, 5.0, 0.0]],
    )


def test_ward_linkage_and_maxclust_cut() -> None:
    profiles = np.asarray([[0.0], [1.0], [2.0]])
    linkage = ward_linkage(euclidean_condensed(profiles), n_samples=3)

    assert linkage.shape == (2, 4)
    np.testing.assert_allclose(linkage[0], [0.0, 1.0, 1.0, 2.0])
    np.testing.assert_allclose(linkage[1, 2], np.sqrt(3.0))

    labels = cut_maxclust(linkage, n_samples=3, n_clusters=2)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_silhouette_samples_precomputed() -> None:
    profiles = np.asarray([[0.0], [0.1], [10.0], [10.1]])
    distances = euclidean_condensed(profiles)
    labels = np.asarray([1, 1, 2, 2], dtype=np.intp)
    scores = silhouette_samples_precomputed(distances, labels)
    assert np.all(scores > 0.9)


def test_hybrid_cut_isolates_low_silhouette_outlier() -> None:
    profiles = np.asarray([[0.0], [0.1], [0.2], [10.0]])
    distances = euclidean_condensed(profiles)
    linkage = ward_linkage(distances, n_samples=4)
    config = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
    )
    labels = HybridOutlierCut().cut(
        linkage,
        distances,
        n_samples=4,
        config=config,
    )
    assert len(np.unique(labels)) == 2
    assert labels[3] != labels[0]
    assert labels[0] == labels[1] == labels[2]


def test_select_cut_strategy_rejects_unknown() -> None:
    with pytest.raises(ProteoForgeValidationError, match="cut 'bad'"):
        select_cut_strategy("bad")
