"""Canonical provenance capture for evaluation runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import warnings


@dataclass(frozen=True)
class GitState:
    commit_sha: str | None
    dirty: bool | None
    available: bool
    limitation: str | None = None


def capture_git_state(path: Path) -> GitState:
    """Capture the exact Git commit and working-tree state containing *path*.

    Archives without ``.git`` metadata are represented explicitly rather than
    inventing provenance.
    """
    path = Path(path).resolve()
    cwd = path if path.is_dir() else path.parent
    try:
        root = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", root, "status", "--porcelain", "--untracked-files=normal"],
            check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return GitState(
            commit_sha=None,
            dirty=None,
            available=False,
            limitation=f"Git provenance unavailable: {type(exc).__name__}",
        )
    dirty = bool(status.strip())
    if dirty:
        warnings.warn("evaluation working tree is dirty", RuntimeWarning, stacklevel=2)
    return GitState(commit_sha=commit, dirty=dirty, available=True)


def require_canonical_git_state(git: GitState, *, allow_dirty_git: bool) -> None:
    """Enforce clean, known Git provenance for a canonical run."""
    if not git.available:
        raise ValueError("canonical run requires available Git provenance")
    if git.dirty and not allow_dirty_git:
        raise ValueError("canonical run requires a clean Git working tree; use explicit override to proceed")
