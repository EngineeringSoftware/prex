#!/usr/bin/env python3
"""
Build a PCP dataset JSONL in parallel by sharding the corpus across workers.

Each worker gets ~1/N of the .imp files (size-balanced), runs the existing
DatasetProcessor create_* entry point on a flat symlink directory, and writes
a shard JSONL. After all workers finish, shards are merged into the final
dataset file and validated.

Usage:
  python scripts/build-pcp-dataset-parallel.py uk sos --workers 8
  python scripts/build-pcp-dataset-parallel.py mk k --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_interpreter.data.dataset_process import (  # noqa: E402
    PCP_EXCLUDED_PROGRAMS,
    DatasetProcessor,
    list_program_files,
)
from llm_interpreter.data.functions import PCP_PROGRAMS_DIR  # noqa: E402
from llm_interpreter.language.language import Language  # noqa: E402
from llm_interpreter.macros import Macros  # noqa: E402

SETUPS = ("uk", "mk")
SEMANTICS = ("sos", "k")
SEMANTICS_TO_TYPE = {
    "sos": Language.SEMANTICS_TYPE.SOS,
    "k": Language.SEMANTICS_TYPE.K,
}
SEMANTICS_TO_EXP = {"sos": "IMP-SOS", "k": "IMP-K"}


def dataset_output_path(setup: str, semantics: str) -> Path:
    exp = SEMANTICS_TO_EXP[semantics]
    return Macros.data_dir / "dataset" / f"dataset-pcp-{setup}-{exp}.jsonl"


def scratch_root(setup: str, semantics: str) -> Path:
    exp = SEMANTICS_TO_EXP[semantics]
    return REPO_ROOT / "scratch" / "pcp-dataset-parallel" / f"{setup}-{exp}"


def expected_records_per_file(setup: str) -> int:
    if setup == "uk":
        return 1
    # mk with include_caucasian_albanian=True -> 2 mutation patterns
    dp = DatasetProcessor(PL="IMP")
    return len(dp._get_mutation_patterns(include_caucasian_albanian=True))


def collect_corpus_files(programs_dir: Path) -> list[Path]:
    files = list_program_files(programs_dir, "IMP")
    return [p for p in files if p.name not in PCP_EXCLUDED_PROGRAMS]


def shard_files_size_balanced(files: list[Path], n_shards: int) -> list[list[Path]]:
    """Assign files to shards by size-descending round-robin."""
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    ranked = sorted(files, key=lambda p: p.stat().st_size, reverse=True)
    shards: list[list[Path]] = [[] for _ in range(n_shards)]
    for index, path in enumerate(ranked):
        shards[index % n_shards].append(path)
    return shards


def prepare_shard_program_dir(programs_dir: Path, files: list[Path]) -> None:
    if programs_dir.exists():
        shutil.rmtree(programs_dir)
    programs_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        dest = programs_dir / src.name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(src.resolve())


def shard_dataset_filename(setup: str, semantics: str) -> str:
    exp = SEMANTICS_TO_EXP[semantics]
    return f"dataset-pcp-{setup}-{exp}.jsonl"


def run_shard_worker(args: tuple) -> dict:
    """
    Worker entry: build one shard JSONL via DatasetProcessor.

    args: (shard_index, setup, semantics, file_paths, scratch_base)
    """
    shard_index, setup, semantics, file_paths, scratch_base = args
    scratch_base = Path(scratch_base)
    file_paths = [Path(p) for p in file_paths]

    shard_root = scratch_base / f"shard-{shard_index}"
    programs_dir = shard_root / "programs"
    out_dir = shard_root / "out"
    shard_tmp = shard_root / "tmp"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    if shard_tmp.exists():
        shutil.rmtree(shard_tmp)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_tmp.mkdir(parents=True, exist_ok=True)

    # Isolate ANTLR / write_to_tmp artifacts so parallel workers do not race on
    # the shared Macros.tmp_dir/IMP_*_ANTLR directory.
    Macros.tmp_dir = shard_tmp

    prepare_shard_program_dir(programs_dir, file_paths)

    semantics_type = SEMANTICS_TO_TYPE[semantics]
    data_processor = DatasetProcessor(PL="IMP")
    started = time.monotonic()

    if setup == "uk":
        data_processor.create_dataset_unmutated_semantics(
            PL="IMP",
            semantics_type=semantics_type,
            task="pcp",
            setup_name="uk",
            raw_programs_dir=programs_dir,
            target_dir=out_dir,
            use_k_framework=False,
        )
    else:
        data_processor.create_dataset_mutated_semantics(
            PL="IMP",
            semantics_type=semantics_type,
            task="pcp",
            setup_name="mk",
            raw_programs_dir=programs_dir,
            target_dir=out_dir,
            include_caucasian_albanian=True,
            use_k_framework=False,
        )

    out_file = out_dir / shard_dataset_filename(setup, semantics)
    if not out_file.is_file():
        raise RuntimeError(
            f"Shard {shard_index} did not produce expected output: {out_file}"
        )

    # Count lines without loading full JSON
    with out_file.open(encoding="utf-8") as handle:
        n_lines = sum(1 for line in handle if line.strip())

    elapsed = time.monotonic() - started
    return {
        "shard_index": shard_index,
        "n_files": len(file_paths),
        "n_records": n_lines,
        "output": str(out_file),
        "elapsed_s": round(elapsed, 2),
    }


def merge_shard_jsonls(shard_paths: list[Path], final_path: Path) -> int:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=final_path.name + ".",
        suffix=".tmp",
        dir=str(final_path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    total = 0
    try:
        with tmp_path.open("w", encoding="utf-8") as out_handle:
            for shard_path in shard_paths:
                with shard_path.open(encoding="utf-8") as in_handle:
                    for line in in_handle:
                        if not line.strip():
                            continue
                        out_handle.write(line if line.endswith("\n") else line + "\n")
                        total += 1
        tmp_path.replace(final_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return total


def check_nonempty_programs(dataset_path: Path) -> None:
    errors: list[str] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_id = record.get("id", f"line-{line_no}")
            for field in ("program", "mutated-program"):
                value = record.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{record_id}: empty or missing {field!r}")
    if errors:
        preview = "\n".join(f"  - {e}" for e in errors[:20])
        more = f"\n  ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise SystemExit(
            f"Nonempty-program check failed ({len(errors)} issue(s)):\n{preview}{more}"
        )
    print("Check passed: all program / mutated-program fields are nonempty.")


def check_corpus_correspondence(
    dataset_path: Path,
    corpus_files: list[Path],
    *,
    records_per_file: int,
) -> None:
    by_name: dict[str, Path] = {}
    for path in corpus_files:
        if path.name in by_name:
            raise SystemExit(
                f"Duplicate corpus filename {path.name}: "
                f"{by_name[path.name]} and {path}"
            )
        by_name[path.name] = path

    counts: Counter[str] = Counter()
    unknown: list[str] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            src = record.get("src-filename")
            record_id = record.get("id", "<unknown>")
            if not isinstance(src, str) or not src:
                unknown.append(f"{record_id}: missing src-filename")
                continue
            if src not in by_name:
                unknown.append(f"{record_id}: src-filename {src!r} not in corpus")
                continue
            counts[src] += 1

    missing = sorted(name for name in by_name if counts[name] == 0)
    wrong_count = sorted(
        (name, counts[name])
        for name in by_name
        if counts[name] != 0 and counts[name] != records_per_file
    )

    problems: list[str] = []
    problems.extend(unknown)
    for name in missing:
        problems.append(f"corpus file {name} missing from dataset")
    for name, count in wrong_count:
        problems.append(
            f"corpus file {name} appears {count} time(s), expected {records_per_file}"
        )

    if problems:
        preview = "\n".join(f"  - {p}" for p in problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise SystemExit(
            f"Corpus correspondence check failed ({len(problems)} issue(s)):\n"
            f"{preview}{more}"
        )
    print(
        f"Check passed: each of {len(by_name)} corpus files appears "
        f"{records_per_file} time(s) in the dataset."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a PCP dataset JSONL in parallel by sharding the "
            "extended-valid-invalid corpus across workers."
        )
    )
    parser.add_argument(
        "setup",
        choices=SETUPS,
        help="Dataset setup: uk (unmutated) or mk (mutated).",
    )
    parser.add_argument(
        "semantics",
        choices=SEMANTICS,
        help="Semantics variant: sos (IMP-SOS) or k (IMP-K).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel shard workers (default: 8).",
    )
    parser.add_argument(
        "--programs-dir",
        type=Path,
        default=None,
        help=(
            "Override PCP programs directory "
            f"(default: {PCP_PROGRAMS_DIR})."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override final dataset JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    programs_dir = (args.programs_dir or PCP_PROGRAMS_DIR).resolve()
    if not programs_dir.is_dir():
        raise SystemExit(f"Programs directory not found: {programs_dir}")

    final_path = (args.output or dataset_output_path(args.setup, args.semantics)).resolve()
    records_per_file = expected_records_per_file(args.setup)
    corpus_files = collect_corpus_files(programs_dir)
    if not corpus_files:
        raise SystemExit(f"No .imp programs found under {programs_dir}")

    n_workers = min(args.workers, len(corpus_files))
    shards = shard_files_size_balanced(corpus_files, n_workers)
    # Drop empty shards if workers > files
    shards = [s for s in shards if s]
    n_workers = len(shards)

    scratch = scratch_root(args.setup, args.semantics)
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    print(
        f"Building dataset-pcp-{args.setup}-{SEMANTICS_TO_EXP[args.semantics]} "
        f"from {len(corpus_files)} programs with {n_workers} worker(s)"
    )
    print(f"Scratch: {scratch}")
    print(f"Output:  {final_path}")
    for i, shard in enumerate(shards):
        bytes_total = sum(p.stat().st_size for p in shard)
        print(f"  shard-{i}: {len(shard)} files, {bytes_total / 1e6:.1f} MB")

    worker_args = [
        (
            i,
            args.setup,
            args.semantics,
            [str(p) for p in shard],
            str(scratch),
        )
        for i, shard in enumerate(shards)
    ]

    started = time.monotonic()
    results: list[dict] = []
    if n_workers == 1:
        results.append(run_shard_worker(worker_args[0]))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(run_shard_worker, wa): wa[0] for wa in worker_args
            }
            for future in as_completed(futures):
                shard_index = futures[future]
                try:
                    result = future.result()
                except Exception as err:
                    raise SystemExit(f"Shard {shard_index} failed: {err}") from err
                results.append(result)
                print(
                    f"  finished shard-{result['shard_index']}: "
                    f"{result['n_records']} records in {result['elapsed_s']}s"
                )

    results.sort(key=lambda r: r["shard_index"])
    shard_paths = [Path(r["output"]) for r in results]
    for path in shard_paths:
        if not path.is_file():
            raise SystemExit(f"Missing shard output: {path}")

    print("Merging shards...")
    n_merged = merge_shard_jsonls(shard_paths, final_path)
    expected_total = len(corpus_files) * records_per_file
    print(
        f"Merged {n_merged} records into {final_path} "
        f"(expected {expected_total})"
    )
    if n_merged != expected_total:
        raise SystemExit(
            f"Merged record count {n_merged} != expected {expected_total}"
        )

    print("\n=== Post-merge checks ===")
    check_nonempty_programs(final_path)
    check_corpus_correspondence(
        final_path,
        corpus_files,
        records_per_file=records_per_file,
    )

    elapsed = time.monotonic() - started
    print(f"\nDone in {elapsed / 60:.1f} min. Dataset: {final_path}")


if __name__ == "__main__":
    main()
