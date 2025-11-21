#!/usr/bin/env python3
import json
import os
import sys
import tempfile
from collections import defaultdict
import multiprocessing as mp
import threading
import logging

# from utils import pdb_utils
import time
from argparse import ArgumentParser
from contextlib import contextmanager
from typing import Any

import numpy as np
import ray


def parse_cmd():
    """Parse command-line arguments.

    Returns
    -------
    Command-line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        type=str,
        help="File that contains a list of protein mutations.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        help="File to which to write the dataset.",
    )
    parser.add_argument(
        "-p",
        "--pdb_dir",
        dest="pdb_dir",
        type=str,
        help="Directory where PDB file are stored.",
    )
    parser.add_argument(
        "-r",
        "--rotations",
        dest="rotations",
        type=float,
        nargs=3,
        help="Rotation angles in radian around all three axes.",
    )
    parser.add_argument(
        "--boxsize",
        dest="boxsize",
        type=int,
        help="Size of the bounding box around the mutation site.",
    )
    parser.add_argument(
        "--voxelsize", dest="voxelsize", type=int, help="Size of the voxel."
    )
    parser.add_argument(
        "-w",
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Whether to overwrite PDBQT files and mutant PDB files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Whether to print verbose messages from HTMD function calls.",
    )
    parser.add_argument(
        "--reverse",
        dest="reverse",
        action="store_true",
        help="This flag indicates that the given list of mutations are reverse mutations",
    )

    # NEW ARGUMENTS
    parser.add_argument(
        "--chain_id",
        dest="chain_id",
        type=str,
        default="A",
        help="Chain identifier in the PDB file. Default is a monomer with chain ID 'A'.",
    )
    parser.add_argument(
        "--num_proc",
        dest="num_proc",
        type=int,
        default=1,
        help="Number of parallel processes to use.",
    )
    parser.add_argument(
        "--cache_dir",
        dest="cache_dir",
        type=str,
        default=None,
        help="Directory to store cached WT feature files. Defaults to <pdb_dir>/feature_cache.",
    )
    # parser.add_argument(
    #     "--no_merge",
    #     dest="no_merge",
    #     action="store_true",
    #     help="If set, do not merge per-group .npy parts into a single output file; keep parts only.",
    # )
    args = parser.parse_args()
    # do any necessary argument checking here before returning
    return args


def _ensure_cache_dir(args):
    """Ensure that the cache directory exists."""
    if args.cache_dir is None:
        cache_dir = os.path.join(args.pdb_dir, "feature_cache")
    else:
        cache_dir = args.cache_dir

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print("Created cache directory:", cache_dir)
    else:
        print("Cache directory already exists:", cache_dir)

    return cache_dir


def main():
    """Main function to generate the dataset."""
    args = parse_cmd()

    print(
        f"Recieved arguments:\n{json.dumps(vars(args), indent=4)}"
    )

    # calculate rotation angles
    if args.rotations is not None:
        rotations = np.pi * np.array(args.rotations)
    else:
        rotations = None

    cache_dir = _ensure_cache_dir(args)
    parts_dir = os.path.join(os.path.dirname(args.output), "parts")
    if not os.path.exists(parts_dir):
        os.makedirs(parts_dir)
        print("Created parts directory:", parts_dir)
    args.parts_dir = parts_dir

    # read all lines, parse into entries
    entries = []
    with open(args.input, "rt") as fh:
        for l in fh:
            if not l.strip():
                continue
            parts = l.strip().split()
            if len(parts) != 4:
                raise ValueError("Each line must be: pdb_chain pos wt mt")
            pdb_chain, pos, wt, mt = parts
            entries.append((pdb_chain, pos, wt, mt))

    print(f"Recieved {len(entries)} entries to process.")

    # group by (pdb_chain, pos)
    groups = defaultdict(list)
    for e in entries:
        key = (e[0], e[1])
        groups[key].append(e)

    # print(f"Grouped into {len(groups)} unique (pdb_chain, pos) pairs.")

    tasks = []
    args_dict = dict(
        pdb_dir=os.path.abspath(args.pdb_dir),
        chain_id=args.chain_id,
        rotations=rotations,
        boxsize=args.boxsize,
        voxelsize=args.voxelsize,
        verbose=args.verbose,
        overwrite=args.overwrite,
        reverse=args.reverse,
        cache_dir=cache_dir,
        parts_dir=parts_dir,
    )

    num_proc = args.num_proc

    for key, group_entries in groups.items():
        tasks.append((key, group_entries, args_dict))

    print(f"Total unique (pdb,pos) groups to process: {len(tasks)}")
    print(f"Using {num_proc} worker(s)")

    # Run tasks with Ray
    all_parts: list[str] = []
    all_meta_files: list[str] = []

    with ray_pool(num_proc=num_proc):
        futures = [process_group.remote(task) for task in tasks]
        all_results = ray.get(futures)
        for r in all_results:
            if not r:
                continue
            for p in r.get("parts", []):
                all_parts.append(p)
            for m in r.get("metadata", []):
                all_meta_files.append(m)

    print(f"Generated {len(all_parts)} part files.")
    print(f"Generated {len(all_meta_files)} metadata files.")

    # DOES NOT MERGE PARTS INTO SINGLE FILE
    # SINCE I ASSUME THAT THIS IS RUN ON MULTIPLE COMPUTERS/NODES


@contextmanager
def ray_pool(num_proc, **ray_init_kwargs):
    """Context manager for Ray pool."""
    try:
        ray.init(
            num_cpus=num_proc,
            **ray_init_kwargs,
        )
        yield
    finally:
        ray.shutdown()


@ray.remote
def process_group(task):
    """Worker function.

    task is a tuple: (group_key, entries, args_dict)
    group_key: (pdb_chain, pos)
    entries: list of tuples (pdb_chain, pos, wt, mt)
    args_dict: simple dict of needed args (pdb_dir, boxsize, voxelsize, ...)
    """

    # Lazy imports so heavy modules and their prints are inside the worker
    # and visible in the remote worker stdout/stderr.
    from utils import pdb_utils  # lazy import

    group_key, entries, args = task
    pdb_chain, pos = group_key
    pdb_dir = args["pdb_dir"]
    chain_id = args["chain_id"]
    rotations = args["rotations"]
    boxsize = args["boxsize"]
    voxelsize = args["voxelsize"]
    verbose = args["verbose"]
    overwrite = args["overwrite"]
    reverse = args["reverse"]
    cache_dir = args["cache_dir"]
    parts_dir = args["parts_dir"]


    wt_pdb_path = os.path.join(pdb_dir, pdb_chain, pdb_chain + "_relaxed.pdb")
    if not os.path.exists(wt_pdb_path):
        raise FileNotFoundError(
            f"PDB file for wild-type does not exist: {wt_pdb_path}")

    wt_cache_file = os.path.join(cache_dir, f"{pdb_chain}_{pos}_wt.npy")

    # Load or compute WT features
    if os.path.exists(wt_cache_file):
        print(
            f"[worker {pdb_chain}:{pos}] Loading WT features from cache: {wt_cache_file}")
        features_wt = np.load(wt_cache_file)
    else:
        print(
            f"[worker {pdb_chain}:{pos}] Computing WT features for: {wt_pdb_path} at pos {pos}")
        features_wt = compute_wt_features(
            pdb_utils,
            pos,
            wt_pdb_path,
            boxsize,
            voxelsize,
            verbose,
            rotations,
            cache_dir,
            wt_cache_file,
        )

    # Remove channel index 6 if present (same behavior as original)
    if features_wt.shape[0] > 6:
        # make the property channels the first axis
        features_wt = np.delete(features_wt, obj=6, axis=0)

    results = []
    metadatas = []

    # Process each mutation in this group
    for entry in entries:
        result = process_each_mutation(
            entry,
            pdb_utils,
            pdb_dir=pdb_dir,
            pdb_chain=pdb_chain,
            pos=pos,
            boxsize=boxsize,
            voxelsize=voxelsize,
            verbose=verbose,
            rotations=rotations,
            reverse=reverse,
            features_wt=features_wt,
        )
        if result is None:
            continue

        _, metadata = result
        metadata["index_in_group"] = len(metadatas)

        results.append(result[0])
        metadatas.append(
            metadata
        )

    # If no results for this group, return empty list (driver will ignore)
    if not results:
        return {"parts": [], "metadata": []}

    # Convert to ndarray and save per-group part file + metadata
    arr = np.array(results)
    # create deterministic filename based on group
    safe_group_name = f"{pdb_chain}_{pos}"
    part_npy = os.path.join(parts_dir, f"{safe_group_name}_part.npy")
    part_meta = os.path.join(parts_dir, f"{safe_group_name}_meta.json")

    # atomic save
    _atomic_save_npy(arr, part_npy)
    _atomic_save_json(metadatas, part_meta)

    print(f"[worker {pdb_chain}:{pos}] Saved {arr.shape[0]} samples to {part_npy}")

    # Return the part filenames and counts so the driver can merge or track
    return {"parts": [part_npy], "metadata": [part_meta]}

def _atomic_save_npy(arr: np.ndarray, target_path: str) -> None:
    """Atomically save numpy array to target_path using temp file + os.replace."""
    tmpf = None
    try:
        fd, tmpf = tempfile.mkstemp(suffix=".npy", dir=os.path.dirname(target_path))
        os.close(fd)
        np.save(tmpf, arr)
        os.replace(tmpf, target_path)
        tmpf = None
    finally:
        if tmpf is not None:
            try:
                os.remove(tmpf)
            except Exception:
                pass


def _atomic_save_json(obj: Any, target_path: str) -> None:
    """Atomically save JSON to target_path."""
    tmpf = None
    try:
        fd, tmpf = tempfile.mkstemp(suffix=".json", dir=os.path.dirname(target_path))
        os.close(fd)
        with open(tmpf, "w") as fh:
            json.dump(obj, fh, indent=2)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except Exception:
                pass
        os.replace(tmpf, target_path)
        tmpf = None
    finally:
        if tmpf is not None:
            try:
                os.remove(tmpf)
            except Exception:
                pass

def process_each_mutation(
        entry: tuple[str, str, str, str],
        pdb_utils: Any,
        *,
        pdb_dir: str,
        pdb_chain: str,
        pos: str,
        boxsize: int,
        voxelsize: int,
        verbose: bool,
        rotations: np.ndarray,
        reverse: bool,
        features_wt: np.ndarray,
) -> None | tuple[np.ndarray, dict]:
    _, pos_e, wt, mt = entry

    mt_pdb_path = os.path.join(
        pdb_dir, pdb_chain, pdb_chain + "_" + wt + pos + mt + "_relaxed.pdb"
    )

    # If the mutant relaxed PDB does not exist, skip this mutation (no Rosetta run).
    if not os.path.exists(mt_pdb_path):
        print(
            f"[worker {pdb_chain}:{pos}] WARNING!!! Mutant relaxed PDB not found, skipping: {mt_pdb_path}")
        return None

    try:
        features_mt = pdb_utils.compute_voxel_features(
            pos_e,
            mt_pdb_path,
            boxsize=boxsize,
            voxelsize=voxelsize,
            verbose=verbose,
            rotations=rotations,
        )
    except Exception as e:
        # If computing mutant features fails for any reason, skip this mutation but report.
        print(
            f"[worker {pdb_chain}:{pos}] Failed to compute MT features for {mt_pdb_path}: {e}")
        return None

    if features_mt.shape[0] > 6:
        # make the property channels the first axis
        features_mt = np.delete(features_mt, obj=6, axis=0)

    if reverse:
        features_combined = np.concatenate((features_mt, features_wt), axis=0)
    else:
        features_combined = np.concatenate((features_wt, features_mt), axis=0)

    return features_combined, {
        "pdb_chain": pdb_chain,
        "group_pos": pos,
        "mut_pos": pos_e,
        "wt": wt,
        "mt": mt,
        "mt_pdb_path": mt_pdb_path,
    }


def compute_wt_features(
        pdb_utils: Any,
        pos: str,
        wt_pdb_path: str,
        boxsize: int,
        voxelsize: int,
        verbose: bool,
        rotations: np.ndarray,
        cache_dir: str,
        wt_cache_file: str,
):
    features_wt = pdb_utils.compute_voxel_features(
        pos,
        wt_pdb_path,
        boxsize=boxsize,
        voxelsize=voxelsize,
        verbose=verbose,
        rotations=rotations,
    )
    print(f"Computed WT features shape: {features_wt.shape}")
    # Persist to cache atomically
    tmpf = None
    try:
        fd, tmpf = tempfile.mkstemp(suffix=".npy", dir=cache_dir)
        os.close(fd)
        np.save(tmpf, features_wt)
        os.replace(tmpf, wt_cache_file)
        tmpf = None
    finally:
        if tmpf is not None:
            try:
                os.remove(tmpf)
            except Exception:
                print(f"Failed to remove temp file: {tmpf}")
                pass

    return features_wt


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed = end_time - start_time
    print("gends-multi.py took", elapsed, "seconds to generate the dataset.")
