"""Apply talker patches to a cloned qwen_megakernel tree.

Idempotent — re-running is a no-op (each edit is guarded by a sentinel).
Anchors are string-based, not line-number-based, so the script tolerates small
upstream drift.

What this patcher does
----------------------

The talker is shape-identical to Qwen3-0.6B except for three things:

  A. **LM head vocab size**: 3,072 (codec tokens) vs 151,936 (text). We make
     ``LDG_VOCAB_SIZE`` build-time overridable so the kernel can be rebuilt
     for either model from the same source tree.

  B. **Composite input embedding**: the talker mixes text-embed + speaker
     embed + reference-audio embed + codec-history + code-group. Doing this
     in CUDA would be a tarpit. Instead, the Python wrapper computes the
     per-step hidden state on GPU in PyTorch and feeds it to the kernel.

     KEY TRICK — **no kernel change needed**: the kernel uses
     ``embed_row = embed_weight + input_token_id * HIDDEN_SIZE`` to fetch the
     first-layer input. If we hand it a single-row "fake embedding table"
     (just our precomputed hidden state, shape [1, HIDDEN_SIZE]) and pass
     ``input_token_id = 0``, the lookup returns our hidden state directly.

  C. **MRoPE sections [24, 20, 20] with rope_theta=1e6**: for pure TTS, only
     the temporal axis is non-trivial; height and width are always 0. That
     means cos/sin entries past the temporal section are 1 / 0 — i.e.
     identity rotation. The kernel's RoPE math is unchanged; we just build
     different precomputed cos/sin tables in Python.

So the only file we actually patch is build.py (and a one-line bit of
kernel.cu to make ``LDG_VOCAB_SIZE`` an overridable macro instead of a
constexpr). Everything else lives in ``talker_model.py``.

Usage::

    python apply_patches.py /path/to/qwen_megakernel
"""

import argparse
import sys
from pathlib import Path


SENTINEL = "QWEN_TTS_TALKER_PATCH_v1"


def _patch_file(path: Path, edits: list[tuple[str, str, str]]) -> int:
    text = path.read_text()
    if SENTINEL in text:
        print(f"  [skip] {path.name} already patched ({SENTINEL})")
        return 0

    applied = 0
    for anchor, old, new in edits:
        if anchor not in text:
            print(f"  [WARN] anchor not found in {path.name}: {anchor!r}", file=sys.stderr)
            continue
        if old not in text:
            print(f"  [WARN] old block not found in {path.name}: {old!r}", file=sys.stderr)
            continue
        text = text.replace(old, new, 1)
        applied += 1

    if applied:
        path.write_text(text)
        print(f"  [ok] {path.name}: {applied} edit(s) applied")
    return applied


def patch_kernel_cu(root: Path) -> None:
    """Patch A — make LDG_VOCAB_SIZE overridable via -D."""
    p = root / "csrc" / "kernel.cu"
    edits: list[tuple[str, str, str]] = [
        (
            "constexpr int LDG_VOCAB_SIZE",
            "// LM head\nconstexpr int LDG_VOCAB_SIZE = 151936;\n",
            (
                "// LM head\n"
                f"// {SENTINEL}: overridable via -DLDG_VOCAB_SIZE for the TTS talker (codec vocab=3072).\n"
                "#ifndef LDG_VOCAB_SIZE\n"
                "#define LDG_VOCAB_SIZE 151936\n"
                "#endif\n"
            ),
        ),
    ]
    _patch_file(p, edits)


def patch_build_py(root: Path) -> None:
    """Patch A (continued) — expose LDG_VOCAB_SIZE as an env-overridable flag.

    Also: when LDG_VOCAB_SIZE is small (talker case), the default 1280-block
    LM head partition is wildly over-provisioned. Halve it when vocab < 16k.
    """
    p = root / "qwen_megakernel" / "build.py"
    edits: list[tuple[str, str, str]] = [
        (
            "KERNEL_FLAGS = [",
            "KERNEL_FLAGS = [\n    f\"-DLDG_NUM_BLOCKS={_env_int('LDG_NUM_BLOCKS', 128)}\",",
            (
                f"# {SENTINEL}: vocab size overridable (text=151936, TTS codec=3072).\n"
                "_VOCAB = _env_int('LDG_VOCAB_SIZE', 151936)\n"
                "_LM_BLOCKS_DEFAULT = 1280 if _VOCAB >= 16384 else 64\n"
                "\n"
                "KERNEL_FLAGS = [\n"
                "    f\"-DLDG_VOCAB_SIZE={_VOCAB}\",\n"
                "    f\"-DLDG_NUM_BLOCKS={_env_int('LDG_NUM_BLOCKS', 128)}\","
            ),
        ),
        (
            "f\"-DLDG_LM_NUM_BLOCKS={_env_int('LDG_LM_NUM_BLOCKS', 1280)}\"",
            "f\"-DLDG_LM_NUM_BLOCKS={_env_int('LDG_LM_NUM_BLOCKS', 1280)}\"",
            "f\"-DLDG_LM_NUM_BLOCKS={_env_int('LDG_LM_NUM_BLOCKS', _LM_BLOCKS_DEFAULT)}\"",
        ),
    ]
    _patch_file(p, edits)


def patch_pyproject(root: Path) -> None:
    """Add a minimal pyproject.toml so ``pip install -e`` works.

    The upstream repo expects you to run from its working directory (no
    setup.py). For our integration we want it installable into the system
    site-packages so other modules can ``from qwen_megakernel.model import ...``.
    """
    p = root / "pyproject.toml"
    if p.exists():
        print("  [skip] pyproject.toml already present")
        return
    p.write_text(
        f"""# {SENTINEL}: minimal pyproject.toml for editable install
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "qwen-megakernel"
version = "0.1.0"
description = "AlpinDale's qwen_megakernel + Qwen3-TTS talker patches"
requires-python = ">=3.10"

[tool.setuptools]
packages = ["qwen_megakernel"]
include-package-data = true

[tool.setuptools.package-data]
qwen_megakernel = ["../csrc/*.cu", "../csrc/*.cpp"]
"""
    )
    print(f"  [ok] {p.name}: created")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="path to a cloned qwen_megakernel/ tree")
    args = ap.parse_args()

    root = args.root.resolve()
    if not (root / "csrc" / "kernel.cu").exists():
        print(f"ERROR: {root} doesn't look like a qwen_megakernel tree", file=sys.stderr)
        return 2

    print(f"Patching {root}")
    patch_kernel_cu(root)
    patch_build_py(root)
    patch_pyproject(root)
    print()
    print("Done. Sanity checks:")
    print("  1. Default Qwen3-0.6B path unchanged:")
    print("       python -m qwen_megakernel.bench")
    print("  2. TTS talker build with codec vocab:")
    print("       LDG_VOCAB_SIZE=3072 python -c 'from qwen_megakernel.build import get_extension; get_extension()'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
