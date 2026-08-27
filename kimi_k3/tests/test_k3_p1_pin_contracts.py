"""P1 / gate G10 -- the IFU tripwire.

K3 is fork-local, so it does not own the mechanisms it depends on. Eight of them
live in `megatron/**` and every one would break *silently* if upstream moved:
a renamed hook would leave the AttnRes payload untransported, a changed
`backward_step` would make the single-tensor packing unnecessary (or
insufficient), a changed `MoELayer.postprocess` would double-apply a norm.

These run on CPU in CI stage 0 and are the first thing re-run after an IFU
rebase (rules R4.5, R10.3). Each failure message says what to re-check, not just
that something changed.
"""

import subprocess
from pathlib import Path

import pytest

from kimi_k3.model.core_patch import PIN_CONTRACTS, assert_pin_contracts

REPO = Path(__file__).resolve().parents[2]
GUARD = REPO / "kimi_k3" / "ci" / "no_core_diff_guard.sh"


@pytest.mark.parametrize("name,fn", PIN_CONTRACTS, ids=[n for n, _ in PIN_CONTRACTS])
def test_pin_contract(name, fn):
    fn()


def test_every_contract_runs_from_the_entry_point():
    assert len(assert_pin_contracts()) == len(PIN_CONTRACTS) == 8


# --- gate G9: the guard itself ----------------------------------------------


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )


@pytest.fixture()
def scratch_repo(tmp_path):
    """A throwaway repo shaped like ours, so the guard is tested hermetically."""
    repo = tmp_path / "repo"
    (repo / "kimi_k3" / "ci").mkdir(parents=True)
    (repo / "megatron" / "core").mkdir(parents=True)
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / "kimi_k3" / "ci" / "no_core_diff_guard.sh").write_bytes(GUARD.read_bytes())
    (repo / "kimi_k3" / "ci" / "no_core_diff_guard.sh").chmod(0o755)
    (repo / "megatron" / "core" / "thing.py").write_text("x = 1\n")

    _git(repo.parent, "init", "-q", str(repo))
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "branch", "-M", "rocm_dev")
    _git(repo, "checkout", "-qb", "feature")
    return repo


def _run_guard(repo, *args):
    return subprocess.run(
        [str(repo / "kimi_k3" / "ci" / "no_core_diff_guard.sh"), *args],
        cwd=repo,
        capture_output=True,
        text=True,
    )


def test_guard_accepts_changes_inside_kimi_k3(scratch_repo):
    (scratch_repo / "kimi_k3" / "new.py").write_text("y = 2\n")
    _git(scratch_repo, "add", "-A")
    r = _run_guard(scratch_repo, "--staged")
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_guard_rejects_a_core_edit(scratch_repo):
    (scratch_repo / "megatron" / "core" / "thing.py").write_text("x = 2\n")
    _git(scratch_repo, "add", "-A")
    r = _run_guard(scratch_repo, "--staged")
    assert r.returncode == 1
    assert "megatron/core/thing.py" in r.stderr
    assert "upstream_proposals" in r.stderr


def test_guard_honours_the_two_file_allowlist(scratch_repo):
    (scratch_repo / ".github" / "workflows" / "kimi_k3.yml").write_text("name: k3\n")
    (scratch_repo / ".pre-commit-config.yaml").write_text("repos: []\n")
    _git(scratch_repo, "add", "-A")
    r = _run_guard(scratch_repo, "--staged")
    assert r.returncode == 0, r.stderr


def test_guard_rejects_another_root_file(scratch_repo):
    """The allowlist is exactly two files, not 'root files in general'."""
    (scratch_repo / "setup.py").write_text("# nope\n")
    _git(scratch_repo, "add", "-A")
    r = _run_guard(scratch_repo, "--staged")
    assert r.returncode == 1
    assert "setup.py" in r.stderr


def test_guard_compares_against_the_merge_base_not_the_tip(scratch_repo):
    """After an IFU rebase the branch carries upstream's core commits legitimately.

    A tip comparison would flag every one of them; a merge-base comparison sees
    only what this branch added.
    """
    # upstream moves core forward
    _git(scratch_repo, "checkout", "-q", "rocm_dev")
    (scratch_repo / "megatron" / "core" / "thing.py").write_text("x = 99\n")
    _git(scratch_repo, "add", "-A")
    _git(scratch_repo, "commit", "-qm", "upstream core change")

    # our branch merges it in, then adds only kimi_k3/ work
    _git(scratch_repo, "checkout", "-q", "feature")
    _git(scratch_repo, "merge", "-q", "--no-edit", "rocm_dev")
    (scratch_repo / "kimi_k3" / "ours.py").write_text("z = 3\n")
    _git(scratch_repo, "add", "-A")
    _git(scratch_repo, "commit", "-qm", "k3 work")

    r = _run_guard(scratch_repo)
    assert r.returncode == 0, f"stdout={r.stdout} stderr={r.stderr}"


def test_guard_is_executable_and_lists_the_documented_allowlist():
    text = GUARD.read_text()
    assert GUARD.stat().st_mode & 0o111, "guard is not executable"
    assert '".github/workflows/kimi_k3.yml"' in text
    assert '".pre-commit-config.yaml"' in text
