#!/usr/bin/env python
"""Convert each Kaggle kernel .py source into a structured .ipynb notebook.

Notebook layout (mirrors the reference notebooks):
    1. markdown  : title + description
    2. code      : imports + logging setup
    3. code      : CONFIG constants
    4. code      : helper functions / path discovery
    5..N         : for each "# ---- Section ----" / "# STEP n/m" block in the
                   source -> markdown header cell ("## n. Section") + code cell

The .py file stays the single source of truth (still runnable as a script);
kernel-metadata.json is updated to point at the notebook.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def discover_kernels():
    """Every kaggle/account* folder that has a kernel-metadata.json + .py source."""
    dirs = []
    for d in sorted(ROOT.glob("account*")):
        if d.is_dir() and (d / "kernel-metadata.json").exists() and list(d.glob("*.py")):
            dirs.append(d.name)
    return dirs


KERNELS = discover_kernels()

# Optional pretty titles for the generated section headers.
SECTION_TITLES = {
    "Load train": "Load train & compact indices",
    "Load train + compact indices": "Load train & compact indices",
    "Build graph": "Build normalized bipartite adjacency",
    "Load factors + SBERT from deep-hybrid kernel output": "Load CF factors + SBERT embeddings",
    "Load factors + SBERT + graph embeddings": "Load CF / graph factors + SBERT embeddings",
    "Load factors + SBERT + graph embeddings from kernel outputs": "Load CF / graph factors + SBERT embeddings",
    "SBERT with real text (same as other kernels)": "Encode SBERT item text",
    "Train + test (same as other kernels)": "Build train / test splits",
    "Test + history": "Build test split & user history",
    "Baselines": "Baseline scores (CF / SBERT / late fusion)",
    "LightGCN": "Train LightGCN (BPR)",
    "Two-Tower": "Train Two-Tower model",
    "Transformer Fusion": "Train Transformer Fusion model",
}

ACRONYMS = {"sbert", "als", "bpr", "cf", "gpu", "cpu", "dcn", "gcn", "faiss",
            "test", "train", "api", "db", "ids", "ram", "mrr", "hr", "ndcg"}


def pretty_title(t: str) -> str:
    t = re.sub(r"^\d+\.\s*", "", t).strip()          # drop leading "3. "
    if t.isupper():
        words = []
        for w in t.split():
            core = w.lower().strip("(),:;.-+/")
            words.append(w if core in ACRONYMS else w.capitalize())
        t = " ".join(words)
    return t[:1].upper() + t[1:] if t else "Step"


def to_source(text: str):
    lines = text.split("\n")
    while lines and lines[-1].strip() == "":
        lines.pop()
    if not lines:
        return []
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def is_section(line: str) -> bool:
    """Top-level comment banners used as section markers in the kernels."""
    if not line.startswith("#"):
        return False
    if re.match(r"^#\s*STEP\b", line):                       # "# STEP 2/5 — ..."
        return True
    if re.match(r"^#\s*\d+\.\s*[A-Za-z]", line):             # "# 1. LOAD + FILTER"
        return True
    if re.match(r"^#\s*-{3,}\s*(\d+\.\s*)?[A-Za-z]", line):   # "# ---- 2. SBERT (real text)"
        return True
    return False


def section_title(line: str) -> str:
    t = re.sub(r"^#[\s\-]+", "", line).strip()
    t = re.sub(r"-{3,}$", "", t).strip()
    t = re.sub(r"^STEP\s+\d+\s*/\s*\d+\s*[—\-:]*\s*", "", t, flags=re.I)
    t = t.strip()
    pretty = pretty_title(t)
    return SECTION_TITLES.get(t, pretty)


def split_sections(src: str):
    """Return (header_lines, [(title, body_lines), ...])."""
    header, sections = [], []
    cur, cur_title = [], None
    for line in src.split("\n"):
        if is_section(line):
            if cur_title is None:
                header = cur
            elif any(x.strip() for x in cur):
                sections.append((cur_title, cur))
            cur, cur_title = [line], section_title(line)
        else:
            cur.append(line)
    if cur_title is None:
        header = cur
    elif any(x.strip() for x in cur):
        sections.append((cur_title, cur))
    return header, sections


def split_header(lines):
    """Split the header block into imports / CONFIG / helpers cells."""
    cfg_start = None
    for i, ln in enumerate(lines):
        if re.match(r"^[A-Z][A-Z0-9_]{2,}\s*=", ln):
            cfg_start = i
            break
    if cfg_start is None:
        return [("\n".join(lines), None)]
    helper_start = None
    for i in range(cfg_start + 1, len(lines)):
        if re.match(r"^(def |class )", lines[i]):
            helper_start = i
            break
    end = helper_start if helper_start is not None else len(lines)
    cells = []
    if "\n".join(lines[:cfg_start]).strip():
        cells.append(("\n".join(lines[:cfg_start]), None))
    if "\n".join(lines[cfg_start:end]).strip():
        cells.append(("\n".join(lines[cfg_start:end]), "CONFIG"))
    if helper_start is not None and "\n".join(lines[end:]).strip():
        cells.append(("\n".join(lines[end:]), None))
    return cells


def md_cell(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": to_source(text)}


def code_cell(text: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": to_source(text),
    }


def build_notebook(title: str, doc: str, src: str):
    doc_lines = [ln.strip() for ln in doc.split("\n") if ln.strip()]
    bullets = "\n".join(f"- {ln}" for ln in doc_lines[:6])
    intro = f"# {title}\n\n{bullets}\n\n**Order:** imports → CONFIG → helpers → steps 1..N (logs in every cell)."
    cells = [md_cell(intro)]

    header, sections = split_sections(src)
    for chunk, _label in split_header(header):
        chunk = chunk.strip("\n")
        if chunk.strip():
            cells.append(code_cell(chunk))

    for i, (sec_title, body_lines) in enumerate(sections, start=1):
        body = "\n".join(body_lines).strip("\n")
        if not body.strip():
            continue
        cells.append(md_cell(f"## {i}. {sec_title}"))
        cells.append(code_cell(body))
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
    m = re.search(r'"""(.+?)"""', src, re.S)
    doc = m.group(1).strip() if m else ""

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
    for d in [ROOT / k for k in KERNELS if not only or k in only]:
        convert(d)


if __name__ == "__main__":
    main()
