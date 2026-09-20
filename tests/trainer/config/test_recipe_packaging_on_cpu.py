"""Keep required recipes in Git/Ray packages without exposing local launchers."""

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
INCLUDED = (
    "recipe/dapo/main_dapo.py",
    "recipe/dapo/main_prime_dapo.py",
    "recipe/dapo/prime_dapo_ray_trainer.py",
    "recipe/dapo/config/dapo_trainer.yaml",
    "recipe/dapo/config/prime_dapo_trainer.yaml",
    "recipe/dapo/config/spo_tree_trainer.yaml",
    "recipe/prime/__init__.py",
    "recipe/prime/prime_dp_rm.py",
    "recipe/prime/prime_fsdp_workers.py",
    "recipe/prime/prime_core_algos.py",
    "recipe/prime/config/prime_trainer.yaml",
)
EXCLUDED = (
    "recipe/dapo/ours.sh",
    "recipe/dapo/prime.sh",
    "recipe/dapo/runtime_env.yaml",
    "recipe/dapo/train.log",
    "recipe/dapo/__pycache__/main_dapo.cpython-312.pyc",
    "recipe/prime/run_prime_qwen.sh",
    "recipe/prime/__pycache__/prime_dp_rm.cpython-312.pyc",
    "recipe/.git.submodule-backup/config",
    "recipe/unrelated/local.py",
)


@pytest.fixture
def packaging_tree(tmp_path):
    shutil.copyfile(ROOT / ".gitignore", tmp_path / ".gitignore")
    for relative in INCLUDED + EXCLUDED:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return tmp_path


def test_git_includes_recipe_dependencies_but_not_local_files(packaging_tree):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    subprocess.run(["git", "init", "-q", str(packaging_tree)], check=True)
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=packaging_tree,
        input="\n".join(INCLUDED + EXCLUDED) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    assert set(result.stdout.splitlines()) == set(EXCLUDED)


def test_ray_can_traverse_recipe_directories(packaging_tree):
    from ray._private.runtime_env.packaging import _dir_travel, _get_ignore_file

    packed = set()

    def collect(path):
        if path.is_file():
            packed.add(path.relative_to(packaging_tree).as_posix())

    # Real Ray traversal prunes entire ignored directories before seeing files.
    _dir_travel(
        packaging_tree / "recipe",
        [_get_ignore_file(packaging_tree, ".gitignore")],
        collect,
        include_gitignore=True,
    )
    assert packed == set(INCLUDED)
