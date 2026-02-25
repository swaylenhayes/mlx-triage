from mlx_triage.config import KnownBug, load_known_bugs, find_bugs_for_model


def test_load_known_bugs():
    bugs = load_known_bugs()
    assert len(bugs) >= 6
    assert all(isinstance(b, KnownBug) for b in bugs)


def test_known_bug_fields():
    bugs = load_known_bugs()
    bug = next(b for b in bugs if b.id == "MLX-001")
    assert bug.title == "Float16 addmm CPU wrong results"
    assert bug.severity == "critical"
    assert "all" in bug.architecture


def test_find_bugs_by_version_and_arch():
    bugs = load_known_bugs()
    matches = find_bugs_for_model(bugs, mlx_version="0.21.0", architecture="llama")
    assert any(b.id == "MLX-001" for b in matches)


def test_find_bugs_current_version():
    bugs = load_known_bugs()
    matches = find_bugs_for_model(bugs, mlx_version="0.25.1", architecture="llama")
    version_bugs = [b for b in matches if b.affected_versions != ["all"]]
    assert not any(b.id == "MLX-001" for b in version_bugs)


def test_find_bugs_filters_by_architecture():
    bugs = load_known_bugs()
    matches = find_bugs_for_model(bugs, mlx_version="0.21.0", architecture="llama")
    assert not any(b.id == "MLX-005" for b in matches)
