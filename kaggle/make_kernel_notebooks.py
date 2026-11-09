#!/usr/bin/env python
"""Convert each Kaggle kernel .py source into a .ipynb notebook and point
kernel-metadata.json at the notebook (kernel_type = notebook).

Why: Kaggle renders notebooks with per-step cells and live tqdm/step logs,
which is easier to follow than a single long script run.

Usage:
    python kaggle/make_kernel_notebooks.py            # all model/sbert kernels
    python kaggle/make_kernel_notebooks.py account7_train_sbert
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

KERNELS = [
    "account7_train_sbert",
    "account6_train_lightgcn",
    "account6_train_light_gated",
    "account6_train_deep_early",
    "account6_train_graph_light_gated",
    "account6_train_two_tower",
    "account6_train_transformer",
]

def to_source(text: str):
    lines = text.split("\n")
    while lines and lines[-1].strip() == "":
        lines.pop()
    if not lines:
        return []
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


# Split points: "# ---- Title ----" section banners and "# STEP n/m ..." lines.
# Pure dashed separator lines are NOT split points (they stay with the next cell).
def is_section(line: str) -> bool:
    if not line.startswith("#"):
        return False
    if re.match(r"^#\s*STEP\b", line):
        return True
    if re.match(r"^#\s*-{3,}", line):
        rest = re.sub(r"^#[\s\-]+", "", line)
        return bool(re.search(r"[A-Za-z0-9]", rest))
    return False


def split_sections(src: str):
    """Split a python file on '# ---- Title ----' / '# STEP ...' markers."""
    cells = []
    cur = []
    for line in src.split("\n"):
        if is_section(line) and any(x.strip() for x in cur):
            cells.append("\n".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        cells.append("\n".join(cur))
    return cells


def md_cell(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": to_source(text),
    }


def code_cell(text: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": to_source(text),
    }


def build_notebook(title: str, subtitle: str, src: str):
    cells = [md_cell(f"# {title}\n\n{subtitle}")]
    for chunk in split_sections(src):
        chunk = chunk.strip("\n")
        if chunk.strip():
            cells.append(code_cell(chunk))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def convert(kernel_dir: Path) -> Path:
    meta_path = kernel_dir / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text())

    py_files = sorted(p for p in kernel_dir.glob("*.py") if p.name != "__init__.py")
    if not py_files:
        raise SystemExit(f"no .py source in {kernel_dir}")
    py_path = py_files[0]

    src = py_path.read_text()
    doc = ""
    m = re.search(r'"""(.+?)"""', src, re.S)
    if m:
        doc = m.group(1).strip()

    nb = build_notebook(meta.get("title", py_path.stem), doc, src)

    nb_path = kernel_dir / (py_path.stem + ".ipynb")
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))

    meta["code_file"] = nb_path.name
    meta["kernel_type"] = "notebook"
    meta["language"] = "python"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"{kernel_dir.name}: {py_path.name} -> {nb_path.name} ({len(nb['cells'])} cells)")
    return nb_path


def main():
    only = sys.argv[1:]
    dirs = [ROOT / d for d in KERNELS if not only or d in only]
    for d in dirs:
        convert(d)


if __name__ == "__main__":
    main()
