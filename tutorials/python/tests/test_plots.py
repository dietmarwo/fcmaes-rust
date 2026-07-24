import json

import matplotlib

matplotlib.use("Agg")

from fcmaes_tutorial_plots import load_run, render_run


def test_render_pareto_qd_and_convergence(tmp_path):
    (tmp_path / "pareto.csv").write_text(
        "point_id,feasible,selected,objective_a,objective_b\n"
        "0,1,1,1.0,3.0\n1,1,0,2.0,2.0\n",
        encoding="utf-8",
    )
    (tmp_path / "qd.csv").write_text(
        "niche_id,grid_x,grid_y,quality_train,quality_validation,"
        "descriptor_a_train,descriptor_b_train,"
        "descriptor_a_validation,descriptor_b_validation\n"
        "0,0,0,2.0,2.2,0.25,0.25,0.27,0.24\n"
        "3,1,1,1.0,1.1,0.75,0.75,0.74,0.77\n",
        encoding="utf-8",
    )
    (tmp_path / "convergence.csv").write_text(
        "evaluations,elapsed_seconds,coverage,qd_score\n"
        "10,0.1,0.25,1.0\n20,0.2,0.5,2.5\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": 1,
        "tutorial": "fixture",
        "formulation": "mo+qd",
        "objectives": [
            {"column": "objective_a", "label": "A"},
            {"column": "objective_b", "label": "B"},
        ],
        "descriptors": [
            {"column": "descriptor_a", "label": "A", "bounds": [0, 1]},
            {"column": "descriptor_b", "label": "B", "bounds": [0, 1]},
        ],
        "qd": {"grid_shape": [2, 2]},
        "convergence_metrics": ["coverage", "qd_score"],
        "artifacts": {
            "pareto": "pareto.csv",
            "qd_archive": "qd.csv",
            "convergence": "convergence.csv",
        },
    }
    (tmp_path / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    rendered = render_run(load_run(tmp_path / "run.json"), tmp_path / "images")
    assert set(rendered) == {
        "pareto",
        "qd_archive",
        "qd_archive_validation",
        "convergence",
    }
    assert all(path.stat().st_size > 1000 for path in rendered.values())
