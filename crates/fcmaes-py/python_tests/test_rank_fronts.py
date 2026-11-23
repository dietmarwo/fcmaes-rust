import numpy as np
import pytest

from fcmaes_rust import (
    ArchiveDE, BordaBoxIndex, GridBoxIndex, ParetoArchive2D, ParetoArchiveND, igd,
)


@pytest.mark.parametrize("make", [BordaBoxIndex, lambda p: GridBoxIndex(p, 0.01)])
def test_closed_boxes_own_input_and_report_exact_batch_ids(make):
    rng = np.random.default_rng(19)
    points = rng.integers(-12, 12, (1024, 6)).astype(float)[:, ::2]
    original = points.copy()
    index = make(points)
    points[:] = 99
    lo = np.array([[-8, -1, 0], [-np.inf] * 3, [np.nan, 0, 0], original[7]])
    hi = np.array([[7, 8, 9], [np.inf] * 3, [1, 1, 1], original[7]])
    offsets, ids, stats = index.query_batch(lo, hi)
    assert offsets.dtype == np.uint64 and ids.dtype == np.uint32
    assert len(stats) == len(lo)
    for i, (a, b) in enumerate(zip(lo, hi)):
        got = ids[offsets[i]:offsets[i+1]]
        np.testing.assert_array_equal(index.query(a, b), got)
        np.testing.assert_array_equal(np.sort(got), np.flatnonzero(((original >= a) & (original <= b)).all(axis=1)))
    with pytest.raises(ValueError):
        index.query([1, 2], [1, 2])
    with pytest.raises(ValueError):
        make([[np.nan]])
    with pytest.raises(ValueError):
        GridBoxIndex(np.zeros((1, 65)), 1.0)


def test_archive_policies_payloads_strides_and_rollback():
    y = np.array([[0., 1.], [0.5, 0.5], [0.5, 0.5], [1., 0.], [3., -1.]])
    x = np.arange(20.).reshape(5, 4)[:, ::2]
    for archive, flags in [(ParetoArchive2D([2., 2.], 2), [1, 1, 0, 1, 0]),
                           (ParetoArchiveND(2, 2), [1, 1, 1, 1, 1])]:
        np.testing.assert_array_equal(archive.add_batch(y, x), np.array(flags, dtype=bool))
        xs, ys = archive.front()
        np.testing.assert_array_equal(xs, x[np.array(flags, dtype=bool)])
        with pytest.raises(ValueError):
            archive.add_batch(y, x[:, 0:1])
        assert len(archive) == sum(flags)
        with pytest.raises(ValueError, match="finite"):
            archive.add_batch([[0.25, 0.4], [np.nan, 2]], x[:2])
        assert any((row == [0.25, 0.4]).all() for row in archive.front()[1])
        assert ys.shape[0] == sum(flags)
        archive.clear()
        assert len(archive) == 0
        assert archive.front()[0].shape == (0, 2)
    hv = ParetoArchive2D([2., 2.])
    assert hv.add([2., 2.]) and hv.hypervolume() == 0.
    with pytest.raises(OverflowError):
        hv.add([-np.finfo(float).max, -np.finfo(float).max])
    assert len(hv) == 1


def test_ask_tell_budget_empty_shapes_invalid_rows_and_distances():
    s = ArchiveDE(2, [-2., 1., 0.], [2., 1., 4.], popsize=8, max_evaluations=19, seed=42)
    assert s.front()[0].shape == (0, 3) and s.front()[1].shape == (0, 2)
    for n in [8, 8, 3]:
        xs = s.ask()
        assert xs.shape == (n, 3) and (xs[:, 1] == 1.).all()
        ys = np.column_stack((xs[:, 0], -xs[:, 0]))
        xs[:] = 100.
        assert np.max(s.ask()) <= 4.
        with pytest.raises(ValueError):
            s.tell(np.full((n, 2), np.nan))
        s.tell(ys)
    assert s.nfev == 19 and s.ask().shape == (0, 3)
    xs, ys = s.front()
    assert len(xs) == 19
    np.testing.assert_array_equal(ys, np.column_stack((xs[:, 0], -xs[:, 0])))
    assert igd([[0., 0.]], [[0., 0.], [3., 4.]]) == 2.5
    with pytest.raises(ValueError):
        igd(np.empty((0, 2)), [[1., 2.]])


@pytest.mark.parametrize("n", [40, 300])
def test_ties_keep_row_order_and_match_the_documented_order(n):
    values = np.array([[-0.0, 0.0, 1.5, -2.0, 1.0][i % 5] for i in range(n)])
    want = np.argsort(values + 0.0, kind="stable")
    lo, hi = [-np.inf], [np.inf]
    np.testing.assert_array_equal(BordaBoxIndex(values[:, None]).query(lo, hi), want)
    np.testing.assert_array_equal(GridBoxIndex(values[:, None], 1.0 / n).query(lo, hi), want)


def test_properties_and_defaults():
    nd = ParetoArchiveND(3, payload_dim=2)
    hv = ParetoArchive2D([1.0, 2.0], payload_dim=4)
    assert (nd.nobj, nd.payload_dim) == (3, 2)
    assert hv.reference == (1.0, 2.0) and hv.payload_dim == 4
    de = ArchiveDE(2, [0.0], [1.0], popsize=4)
    first = de.ask()
    assert first.shape == (4, 1)
    np.testing.assert_array_equal(first, ArchiveDE(2, [0.0], [1.0], popsize=4, seed=0).ask())
    de.tell(np.column_stack((first[:, 0], 1.0 - first[:, 0])))
    assert de.nfev == 4
