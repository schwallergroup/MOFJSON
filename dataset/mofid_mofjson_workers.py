"""Top-level worker functions for cif_to_mofid_mofjson_batch.ipynb.

These live in a real, importable .py file -- not inside notebook cells --
deliberately. Python's `multiprocessing` needs a fresh, isolated process per
CIF here (see `_run_isolated`'s docstring for why), and this module forces
the 'spawn' start method explicitly so behavior is identical on Linux and
macOS. 'spawn' pickles the target by its import path, not its bytecode --
a function/closure defined inside a Jupyter cell has no such import path and
would fail (or silently misbehave) under spawn. Keeping the real per-CIF
work here, with the notebook only importing and calling it, is what makes
the hard-timeout wrapper actually portable to your Mac.

MOFid note: this does NOT use the official Python `mofid` package (which
needs a compiled Zeo++/OpenBabel/Java toolchain). It instead shells out to
`compute_mofid.cjs` in the `mofid_web_assets/` folder delivered alongside
this file -- a headless Node.js run of the EXACT SAME WASM+JS assets that
power the official web tool at
https://snurr-group.github.io/web-mofid/sbu.html (verified directly: same
node/linker SMILES fragments, same RCSR topology symbol, for the same real
CIF, as this project's own independently-computed MOFJSON topology). One
process per CIF is deliberate here too, not just for timeout safety: an
attempt to reuse one warm Node process across multiple CIFs (to amortize
the ~7s module-load cost) produced a WRONG result on the second CIF in the
same process, most likely static/global state inside the compiled C++ code
that assumes one conversion per process lifetime -- the same assumption
the web page itself relies on across a full page reload. One fresh process
per CIF is the verified-safe design; each costs about that same ~7-8s
fixed overhead regardless, which is why every one of these still needs its
own `_run_isolated` call.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import subprocess
import warnings
from pathlib import Path
from typing import Any

_CTX = mp.get_context("spawn")


def _mofid_target(cif_path: str, mofid_assets_dir: str, queue) -> None:
    try:
        script_path = str(Path(mofid_assets_dir) / "compute_mofid.cjs")
        proc = subprocess.run(
            ["node", script_path, cif_path],
            capture_output=True,
            text=True,
        )
        stdout = proc.stdout.strip()
        if not stdout:
            queue.put((
                "error",
                f"node produced no stdout (exit={proc.returncode}): {proc.stderr[-2000:]}",
            ))
            return
        # compute_mofid.cjs always writes exactly one JSON line, success or
        # failure -- see that script's own header for why it's designed
        # this way (so a caller can always json.loads(stdout) without
        # checking the exit code first).
        payload = json.loads(stdout.splitlines()[-1])
        if "error" in payload:
            queue.put(("error", payload["error"]))
            return
        queue.put(("ok", payload))
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: this is a
        # batch job driving research-grade C++/WASM code we don't control.
        # ANY exception here should become one logged, skippable failure for
        # THIS structure, not a crash of the whole 2000-structure run.
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _mofjson_target(cif_path: str, project_root: str, output_path: str, queue) -> None:
    try:
        import sys

        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from pymatgen.io.cif import CifParser

        from src.mofjson import AssignRoles, BuildDocument, RawStructure, StructureToJSON

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = CifParser(cif_path).parse_structures(primitive=False)[0]

        mofjson_flat = StructureToJSON(structure, include_empty_fields=True)
        node_linker_roles = AssignRoles(mofjson_flat["atoms"], mofjson_flat["bonds"])
        raw = RawStructure(
            lattice=mofjson_flat["lattice"],
            atoms=mofjson_flat["atoms"],
            bonds=mofjson_flat["bonds"],
            charges=mofjson_flat.get("charges") or None,
            aromatic_n_h=mofjson_flat.get("aromatic_n_h") or None,
        )
        doc = BuildDocument(raw, node_linker_roles)
        Path(output_path).write_text(doc.model_dump_json(indent=2))
        queue.put((
            "ok",
            {
                "mofjson_path": output_path,
                "n_building_blocks": len(doc.derived.building_blocks),
                "net": doc.derived.topology.get("net"),
            },
        ))
    except Exception as exc:  # noqa: BLE001 -- same reasoning as above.
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_isolated(target, args: tuple, timeout_seconds: float) -> dict[str, Any]:
    """Run `target(*args, queue)` in its own fresh process; hard-kill on
    timeout.

    A brand-new process per call (not a persistent worker pool) is
    deliberate: mofid's own SBU-extraction step shells out to a compiled
    C++ binary, and its topology step shells out to Java/Systre -- if
    either wedges on a pathological structure (both are documented to hang
    on certain inputs; see mofid's own SYSTRE_TIMEOUT), terminating THIS
    process reliably reclaims it without needing the child to cooperate.
    A shared worker pool could not offer that guarantee -- a wedged task
    would just occupy a pool slot forever."""
    queue = _CTX.Queue()
    proc = _CTX.Process(target=target, args=(*args, queue))
    proc.start()
    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {"status": "timeout", "error": f"exceeded {timeout_seconds}s"}
    if not queue.empty():
        kind, payload = queue.get()
        if kind == "ok":
            return {"status": "ok", "result": payload}
        return {"status": "error", "error": payload}
    return {"status": "error", "error": f"worker process exited (code={proc.exitcode}) with no result"}


def run_mofid_on_cif(cif_path: str, mofid_assets_dir: str, timeout_seconds: float) -> dict[str, Any]:
    """Compute a real MOFid/MOFkey for one CIF by driving the headless
    Node.js `compute_mofid.cjs` script (see module docstring), isolated in
    its own process with a hard wall-clock timeout. Requires only Node.js
    and the `mofid_web_assets/` folder delivered alongside this file --
    no compiled Zeo++/OpenBabel/Java toolchain needed."""
    return _run_isolated(_mofid_target, (cif_path, mofid_assets_dir), timeout_seconds)


def run_mofjson_on_cif(
    cif_path: str, project_root: str, output_path: str, timeout_seconds: float
) -> dict[str, Any]:
    """Build a real MOFJSON document for one CIF via this project's own
    `BuildDocument` pipeline, isolated the same way as `run_mofid_on_cif`
    for a consistent, resumable batch loop."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return _run_isolated(_mofjson_target, (cif_path, project_root, output_path), timeout_seconds)