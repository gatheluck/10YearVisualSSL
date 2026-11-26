"""Create an independent Git repository and stage its commit as a gitlink."""
from pathlib import Path
import subprocess


def local_gitlink(parent: Path, relative: str, files: dict[str, str], *, env=None) -> str:
    child = parent / relative

    def git(repo, *args):
        return subprocess.check_output(
            ["git", "-c", "maintenance.autoDetach=false", "-c", "gc.autoDetach=false",
             "-C", str(repo), *args], env=env, text=True,
            stderr=subprocess.PIPE).strip()

    child.mkdir(parents=True)
    git(child, "init", "-q")
    for name, content in files.items():
        path = child / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    git(child, "add", ".")
    git(child, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "Pinned dependency")
    revision = git(child, "rev-parse", "HEAD")
    section = f"submodule.{relative}"
    git(parent, "config", "-f", ".gitmodules", section + ".path", relative)
    git(parent, "config", "-f", ".gitmodules", section + ".url", "./unused.git")
    git(parent, "add", ".gitmodules")
    git(parent, "update-index", "--add", "--cacheinfo", f"160000,{revision},{relative}")
    return revision
