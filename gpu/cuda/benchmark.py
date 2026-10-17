#!/usr/bin/env python3
"""Compare exact grouped and direct INT8 matrix stages, never submit mining work."""
import argparse
import ctypes as ct
import datetime as dt
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

METHODS = (
    "direct_cublas", "grouped_cublas_residual", "packed_direct_cublas",
    "direct_mma16_fused", "grouped_mma16_prepared", "grouped_mma16_inline",
    "grouped_mma16_inline_padded128",
)
DIRECT_METHODS = (METHODS[0], METHODS[2], METHODS[3])
GROUPED_METHODS = (METHODS[4], METHODS[5])
EXCLUSIONS = [
    "input commitments", "seed expansion and noise generation", "candidate generation",
    "final keyed BLAKE3 hashes", "winning-proof generation and verification",
    "network submission, rejects, fees and payouts", "useful inference",
    "allocation, compilation and initial fixture transfer to GPU",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words(root, fixture, key, dtype, shape):
    path = (root / fixture[key]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("fixture path escapes root")
    dtype = np.dtype(dtype)
    if path.stat().st_size != int(np.prod(shape)) * dtype.itemsize:
        raise ValueError(f"{fixture['name']}: wrong size for {key}")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def plan_for(pairs):
    """Independent host plan; its active basis must match the Rust exporter."""
    chunks, r, _ = pairs.shape
    offsets, edges, vertices, anchors, dims = [], [], [], [], []
    for c in range(chunks):
        parents = list(range(r))
        incident = [[] for _ in range(r)]

        def root(x):
            while parents[x] != x:
                parents[x] = parents[parents[x]]
                x = parents[x]
            return x

        for s, (a, b) in enumerate(pairs[c]):
            a, b = int(a), int(b)
            x, y = root(a), root(b)
            parents[max(x, y)] = min(x, y)
            incident[a].append(s + 1)
            incident[b].append(-s - 1)
        cv, ca, co = [], [], []
        for v in range(r):
            anchor = root(v)
            if v == anchor:
                continue
            cv.append(v)
            ca.append(anchor)
            co.append(len(edges))
            edges.extend(incident[v])
        dims.append(len(cv))
        offsets.extend(co + [len(edges)] * (r + 1 - len(co)))
        vertices.extend(cv + [0] * (r - len(cv)))
        anchors.extend(ca + [0] * (r - len(ca)))
    return {
        "offsets": offsets, "edges": edges, "vertices": vertices,
        "anchors": anchors, "dims": dims,
    }


def load_fixture(root, f):
    m, n, k, r = (int(f[key]) for key in ("m", "n", "k", "r"))
    if (k, r, f["h"], f["w"]) != (2048, 128, 8, 32) or m <= 0 or n <= 0 or m % 8 or n % 32:
        raise ValueError("unsupported fixture shape")
    chunks = k // r
    specs = {
        "a": ("i1", (m, k)), "bt": ("i1", (n, k)), "v": ("i1", (n, r)),
        "pairs": ("<u4", (chunks, r, 2)), "vertices": ("<u4", (chunks, r)),
        "anchors": ("<u4", (chunks, r)), "d": ("<u4", (chunks,)),
        "prefixes": ("<i4", (chunks, m, n)),
        "states": ("<u4", ((m // 8) * (n // 32), 16)),
        "hashes": ("u1", ((m // 8) * (n // 32), 32)),
    }
    data = {key: words(root, f, key, dtype, shape) for key, (dtype, shape) in specs.items()}
    if np.any(data["pairs"] >= r) or np.any(data["v"] < -32) or np.any(data["v"] > 31):
        raise ValueError("invalid noise inputs")
    return data


def states_from_prefixes(prefixes):
    chunks, m, n = prefixes.shape
    tiled = prefixes.view(np.uint32).reshape(chunks, m // 8, 8, n // 32, 32)
    reduced = np.bitwise_xor.reduce(np.bitwise_xor.reduce(tiled, axis=4), axis=2)
    states = np.zeros((m // 8, n // 32, 16), dtype=np.uint32)
    for c in range(chunks):
        old = states[:, :, c % 16]
        states[:, :, c % 16] = (old << 13) ^ (old >> 19) ^ reduced[c]
    return states.reshape(-1, 16)


def validate_host(f, data, plan_checker=None, root=None):
    m, n, k, r = (f[key] for key in ("m", "n", "k", "r"))
    p = plan_for(data["pairs"])
    for key, fixture_key in (("vertices", "vertices"), ("anchors", "anchors"), ("dims", "d")):
        np.testing.assert_array_equal(np.array(p[key]).reshape(data[fixture_key].shape), data[fixture_key])
    if plan_checker:
        native = json.loads(subprocess.check_output([str(plan_checker), str(root / f["pairs"])], text=True))
        if native != p:
            raise AssertionError(f"{f['name']}: C++ plan differs from Python/Rust")
    a = data["a"].astype(np.int32)
    v = data["v"].astype(np.int32)
    acc = np.zeros((m, n), dtype=np.int32)
    clipped_count = 0
    for c in range(k // r):
        pairs = data["pairs"][c]
        reconstructed_b = v[:, pairs[:, 0]] - v[:, pairs[:, 1]]
        np.testing.assert_array_equal(reconstructed_b, data["bt"][:, c * r:(c + 1) * r])
        d = p["dims"][c]
        g = np.zeros((m, r), dtype=np.int32)
        for t in range(d):
            start, end = p["offsets"][c * (r + 1) + t:c * (r + 1) + t + 2]
            for edge in p["edges"][start:end]:
                g[:, t] += (1 if edge > 0 else -1) * a[:, c * r + abs(edge) - 1]
        lo = np.clip(g, -128, 127)
        residual = g - lo
        clipped_count += int(np.count_nonzero(residual))
        vertices, anchors = data["vertices"][c], data["anchors"][c]
        b = (v[:, vertices] - v[:, anchors]).T.copy()
        b[d:, :] = 0
        # Both stages use integer arithmetic, including all sparse corrections.
        delta = lo @ b + residual @ b
        np.testing.assert_array_equal(delta, a[:, c * r:(c + 1) * r] @ reconstructed_b.T)
        acc += delta
        np.testing.assert_array_equal(acc, data["prefixes"][c])
    np.testing.assert_array_equal(states_from_prefixes(data["prefixes"]), data["states"])
    return {"prefix_entries": int(data["prefixes"].size), "tile_states": int(len(data["states"])),
            "nonzero_residuals": clipped_count, "active_dimensions": p["dims"]}


class Native:
    def __init__(self, path):
        self.lib = ct.CDLL(str(path.resolve()))
        ptr, integer = ct.c_void_p, ct.c_int
        for name, args, result in (
            ("pfm_error", [], ct.c_char_p),
            ("pfm_create", [integer] * 4 + [ptr] * 3, ptr),
            ("pfm_destroy", [ptr], None),
            ("pfm_prepare", [ptr, integer], integer),
            ("pfm_run", [ptr, integer, integer, integer], integer),
            ("pfm_read", [ptr, ptr, ptr], integer),
            ("pfm_measure", [ptr, integer, integer, ptr, ptr], integer),
            ("pfm_device", [ptr, integer, ptr, ptr, ptr, ptr], integer),
        ):
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result

    def check(self, result):
        if result != 0:
            raise RuntimeError(self.lib.pfm_error().decode())

    def device(self):
        name = ct.create_string_buffer(256)
        major, minor, runtime, blas = (ct.c_int() for _ in range(4))
        self.check(self.lib.pfm_device(name, len(name), ct.byref(major), ct.byref(minor), ct.byref(runtime), ct.byref(blas)))
        return {"name": name.value.decode(), "compute_capability": f"{major.value}.{minor.value}",
                "cuda_runtime": runtime.value, "cublas_version": blas.value}

    def measure(self, ctx, method, mode):
        wall, stream = ct.c_double(), ct.c_double()
        self.check(self.lib.pfm_measure(ctx, method, mode, ct.byref(wall), ct.byref(stream)))
        return {"wall_us": wall.value, "cuda_event_span_us": stream.value}


def benchmark_family(native, candidates, loaded, rounds, warmups):
    f = candidates[0]
    inputs = np.stack([data["a"] for data in loaded])
    v = np.ascontiguousarray(loaded[0]["v"])
    pairs = np.ascontiguousarray(loaded[0]["pairs"])
    ctx = native.lib.pfm_create(f["m"], f["n"], f["k"], len(inputs), inputs.ctypes.data, v.ctypes.data, pairs.ctypes.data)
    if not ctx:
        raise RuntimeError(native.lib.pfm_error().decode())
    try:
        for method, name in enumerate(METHODS):
            native.check(native.lib.pfm_prepare(ctx, method))
            for index, (candidate, data) in enumerate(zip(candidates, loaded)):
                prefixes = np.empty(data["prefixes"].shape, dtype=np.int32)
                states = np.empty(data["states"].shape, dtype=np.uint32)
                native.check(native.lib.pfm_run(ctx, method, index, 1))
                native.check(native.lib.pfm_read(ctx, prefixes.ctypes.data, states.ctypes.data))
                np.testing.assert_array_equal(prefixes, data["prefixes"], err_msg=f"{candidate['name']} {name} prefixes")
                np.testing.assert_array_equal(states, data["states"], err_msg=f"{candidate['name']} {name} states")
                # Also validate the timed path, which omits full-prefix writes.
                native.check(native.lib.pfm_run(ctx, method, index, 0))
                native.check(native.lib.pfm_read(ctx, None, states.ctypes.data))
                np.testing.assert_array_equal(states, data["states"], err_msg=f"{candidate['name']} {name} timed states")
            for _ in range(warmups):
                native.measure(ctx, method, 0)
                native.measure(ctx, method, 1)
        raw = []
        samples = {name: [] for name in METHODS}
        for round_id in range(rounds):
            for offset in range(len(METHODS)):
                method = (round_id + offset) % len(METHODS)
                prep = native.measure(ctx, method, 0)
                batch = native.measure(ctx, method, 1)
                total = (prep["wall_us"] + batch["wall_us"]) / len(inputs)
                samples[METHODS[method]].append(total)
                raw.append({"round": round_id, "order": offset, "method": METHODS[method],
                            "b_preparation": prep, "candidate_batch": batch,
                            "wall_us_per_candidate_including_amortized_b": total})
        medians = {name: statistics.median(values) for name, values in samples.items()}
        baseline = min(DIRECT_METHODS, key=medians.get)
        grouped = min(GROUPED_METHODS, key=medians.get)
        ratio = medians[baseline] / medians[grouped]
        paired_wins = sum(samples[grouped][i] < min(samples[name][i] for name in DIRECT_METHODS) for i in range(rounds))
        candidate_medians = {name: statistics.median(x["candidate_batch"]["wall_us"] / len(inputs) for x in raw if x["method"] == name) for name in METHODS}
        prep_medians = {name: statistics.median(x["b_preparation"]["wall_us"] for x in raw if x["method"] == name) for name in METHODS}
        reuse_models = {str(reuse): {name: statistics.median((x["candidate_batch"]["wall_us"] / len(inputs)) + x["b_preparation"]["wall_us"] / reuse for x in raw if x["method"] == name) for name in METHODS} for reuse in (4, 64, 1024)}
        return {"m": f["m"], "n": f["n"], "job_key_id": f["id"],
                "candidate_count": len(inputs), "candidate_names": [c["name"] for c in candidates],
                "median_wall_us_per_candidate": medians, "faster_direct_control": baseline,
                "selected_grouped_method": grouped,
                "grouped_speedup_vs_faster_direct": ratio,
                "all_method_ratios_vs_faster_direct": {name: medians[baseline] / medians[name] for name in METHODS},
                "median_candidate_wall_us": candidate_medians,
                "median_b_preparation_wall_us": prep_medians,
                "modeled_amortized_wall_us_by_reuse": reuse_models,
                "reuse_model_scope": "Costs recombined from measured four-candidate batches, not executions of 64 or 1024 distinct candidates.", "grouped_paired_round_wins": paired_wins,
                "rounds": rounds, "raw_samples": raw}
    finally:
        native.lib.pfm_destroy(ctx)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("--library", type=Path, default=Path(__file__).with_name("libpearl_fast_matmul.so"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", help="Select MxN, e.g. 128x128; omit for all shapes")
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true", help="Host arithmetic checks; no CUDA required")
    parser.add_argument("--plan-checker", type=Path, help="Optional compiled C++ host plan checker")
    args = parser.parse_args()
    if args.rounds < 3 or args.warmups < 1:
        parser.error("at least 3 rounds and 1 warmup required")
    root = args.fixtures.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1 or manifest.get("byte_order") != "little":
        raise ValueError("unsupported manifest format")
    selected = [f for f in manifest["fixtures"] if not args.shape or f"{f['m']}x{f['n']}" == args.shape]
    if not selected or len({f["name"] for f in selected}) != len(selected):
        raise ValueError("empty selection or duplicate fixture names")
    families = {}
    checked = []
    native = None if args.validate_only else Native(args.library)
    device = None if native is None else native.device()
    for f in selected:
        data = load_fixture(root, f)
        if args.validate_only:
            result = validate_host(f, data, args.plan_checker, root)
        else:
            np.testing.assert_array_equal(states_from_prefixes(data["prefixes"]), data["states"])
            result = {"prefix_entries": int(data["prefixes"].size), "tile_states": int(len(data["states"]))}
        checked.append({"name": f["name"], **result})
        families.setdefault((f["m"], f["n"], f["id"]), []).append((f, data))
        print(f"Loaded {f['name']}", flush=True)
    reports = []
    for key in sorted(families):
        family = sorted(families[key], key=lambda item: item[0]["name"])
        candidates, loaded = zip(*family)
        for field in ("v", "pairs", "bt"):
            for data in loaded[1:]:
                np.testing.assert_array_equal(loaded[0][field], data[field], err_msg="B cache cannot be shared")
        for field in ("a", "hashes"):
            fingerprints = {hashlib.sha256(data[field].tobytes()).hexdigest() for data in loaded}
            if len(fingerprints) != len(loaded):
                raise ValueError(f"family {key}: repeated {field}")
        if len({f["seed_a"] for f in candidates}) != len(candidates):
            raise ValueError("A seeds must differ")
        if native:
            result = benchmark_family(native, candidates, loaded, args.rounds, args.warmups)
            reports.append(result)
            print(f"PASS {key}: grouped / faster direct throughput ratio {result['grouped_speedup_vs_faster_direct']:.4f}x", flush=True)
    report = {
        "schema_version": 2, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "host_arithmetic_validated" if native is None else "cuda_matrix_stage_validated_and_benchmarked",
        "device": device, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "manifest_sha256": digest(manifest_path),
        "source_sha256": {p.name: digest(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in (".cu", ".cuh", ".hpp", ".cpp", ".py", ".sh")},
        "fixture_sha256": {f["name"]: {key: digest(root / f[key]) for key in ("a", "bt", "v", "pairs", "prefixes", "states", "hashes")} for f in selected},
        "library_sha256": None if native is None else digest(args.library),
        "checks": checked, "cases": reports,
        "methods": list(METHODS), "excluded": EXCLUSIONS,
        "scope": "Synthetic fixed-B fixtures; finite candidates replayed across timing rounds. Not a miner or profit measurement.",
        "baseline_scope": "Faster of direct cuBLAS, packed direct cuBLAS and fused direct MMA16. Not Pearl's fused NVIDIA kernel.",
        "direct_methods": list(DIRECT_METHODS), "primary_grouped_methods": list(GROUPED_METHODS),
        "warmups": args.warmups,
        "mma16_shape": {"block_m": 16, "block_n": 128, "threads": 128, "instruction": "m16n8k16.s32.s8.s8.s32"},
        "grouped_mma16_dimensions": {f["name"]: [((int(d)+15)//16)*16 for d in load_fixture(root,f)["d"]] for f in selected},
        "legacy_grouped_dense_inner_dimension": 128,
        "mma16_grouped_inner_dimension": "ceil(active_dimension/16)*16; padded control uses 128",
        "cuda_event_scope": "Elapsed stream span; may include host launch/copy gaps, not active GPU time.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
