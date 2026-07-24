import json

import numpy as np
import pytest

from fcmaes_tutorial_plots import load_run, pareto_from_arrays, qd_from_archive


def test_load_run_and_numeric_csv(tmp_path):
    (tmp_path / "pareto.csv").write_text(
        "point_id,objective_a,label\n0,1.5,first\n", encoding="utf-8"
    )
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tutorial": "fixture",
                "formulation": "mo",
                "artifacts": {"pareto": "pareto.csv"},
            }
        ),
        encoding="utf-8",
    )
    run = load_run(tmp_path / "run.json")
    assert run.table("pareto")["objective_a"][0] == 1.5
    assert run.table("pareto")["label"][0] == "first"


def test_load_run_rejects_schema_and_path_escape(tmp_path):
    (tmp_path / "run.json").write_text(
        '{"schema_version":2,"tutorial":"x","formulation":"mo","artifacts":{}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema version"):
        load_run(tmp_path / "run.json")
    (tmp_path / "run.json").write_text(
        '{"schema_version":1,"tutorial":"x","formulation":"mo",'
        '"artifacts":{"pareto":"../outside.csv"}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes"):
        load_run(tmp_path / "run.json").artifact_path("pareto")


def test_pareto_array_adapter_marks_constraints():
    table = pareto_from_arrays(
        [[0.0], [1.0]],
        [[1.0, -0.1], [2.0, 0.2]],
        objective_names=["objective_cost"],
        constraint_count=1,
    )
    np.testing.assert_array_equal(table["feasible"], [1.0, 0.0])


def test_qd_archive_adapter_omits_empty_niches():
    class Archive:
        def ys(self):
            return np.array([1.0, np.inf, 2.0])

        def xs(self):
            return np.array([[1.0], [2.0], [3.0]])

        def descriptors(self):
            return np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    table = qd_from_archive(Archive())
    np.testing.assert_array_equal(table["niche_id"], [0, 2])
    np.testing.assert_allclose(table["quality_train"], [1.0, 2.0])
