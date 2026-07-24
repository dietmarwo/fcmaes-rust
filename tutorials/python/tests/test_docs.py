from pathlib import Path

from check_docs import is_tracked, local_targets


def test_local_targets_skip_remote_and_keep_relative_files(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "[local](data/result.csv)\n"
        "![image](images/plot.svg#panel)\n"
        "[remote](https://example.com/result.csv)\n",
        encoding="utf-8",
    )
    targets = dict(local_targets(readme))
    assert targets["data/result.csv"] == (tmp_path / "data/result.csv").resolve()
    assert targets["images/plot.svg#panel"] == (tmp_path / "images/plot.svg").resolve()
    assert len(targets) == 2


def test_tracked_file_and_directory_detection(tmp_path: Path):
    result = tmp_path / "results" / "run.json"
    result.parent.mkdir()
    result.write_text("{}", encoding="utf-8")
    tracked = {"results/run.json"}
    assert is_tracked(result, tmp_path, tracked)
    assert is_tracked(result.parent, tmp_path, tracked)
    assert not is_tracked(tmp_path / "missing.csv", tmp_path, tracked)
