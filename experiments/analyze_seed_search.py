#!/usr/bin/env python3
"""Validate complete search logs and summarize arithmetic counts, never GPU speed."""
import argparse
import csv
import gzip
import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def analyze(directory, source_dir, output):
    screen_path = directory / 'screen.json'
    screen = json.loads(screen_path.read_text())
    assert (screen['m'], screen['n'], screen['k'], screen['r']) == (2048, 512, 2048, 128)
    assert len(screen['jobs']) == 4
    jobs = []
    for job in screen['jobs']:
        csv_path = directory / f"job-{job['job_id']}-trials.csv"
        distribution = Counter()
        chunk_distribution = Counter()
        scalar_distribution = Counter()
        records = []
        positions, roots, seeds = set(), set(), set()
        with csv_path.open(newline='') as f:
            for expected_trial, row in enumerate(csv.DictReader(f)):
                trial = int(row['trial'])
                pos = int(row['flat_b_position'])
                dims = list(map(int, row['chunk_dimensions'].split(';')))
                scalar = int(row['scalar_inner_sum'])
                padded = int(row['mma16_inner_sum'])
                assert trial == expected_trial
                assert pos == trial * 1_000_003 % (512 * 2048)
                assert pos not in positions and row['root_b'] not in roots and row['seed_b'] not in seeds
                positions.add(pos)
                roots.add(row['root_b'])
                seeds.add(row['seed_b'])
                assert len(dims) == 16 and all(1 <= d <= 127 for d in dims)
                assert scalar == sum(dims)
                assert padded == sum(((d + 15) // 16) * 16 for d in dims)
                records.append((padded, scalar, trial))
                distribution[padded] += 1
                scalar_distribution[scalar] += 1
                chunk_distribution.update(dims)
        assert len(records) == job['trials']
        assert min(records) == (job['best_padded']['mma16_inner_sum'], job['best_padded']['scalar_inner_sum'], job['best_padded']['trial'])
        assert min(records, key=lambda r: (r[1], r[0], r[2])) == (job['best_scalar']['mma16_inner_sum'], job['best_scalar']['scalar_inner_sum'], job['best_scalar']['trial'])
        for history in job['history']:
            subset = records[:history['trials']]
            assert min(subset) == (history['best_padded']['mma16_inner_sum'], history['best_padded']['scalar_inner_sum'], history['best_padded']['trial'])
            assert min(subset, key=lambda r: (r[1], r[0], r[2])) == (history['best_scalar']['mma16_inner_sum'], history['best_scalar']['scalar_inner_sum'], history['best_scalar']['trial'])
        checks = []
        for ref in job['checks']:
            path = directory / ref['file']
            check = json.loads(path.read_text())
            assert check['job_key'] == job['job_key']
            assert check['prefix_entries_checked'] == 4 * 2 * 16 * 8 * 32
            assert check['raw_product_zero']
            assert len({c['root_a'] for c in check['cases']}) == 4
            assert len({c['seed_a'] for c in check['cases']}) == 4
            assert len({t['hash'] for c in check['cases'] for t in c['tiles']}) == 8
            if check['selected']['flat_b_position'] is not None:
                assert check['correction_negative_controls'] == 4
            checks.append({**ref, 'sha256': sha256(path)})
        compressed = csv_path.with_suffix('.csv.gz')
        with csv_path.open('rb') as f, compressed.open('wb') as out:
            with gzip.GzipFile(filename=csv_path.name, mode='wb', fileobj=out, mtime=0) as gz:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    gz.write(block)
        baseline = job['baseline']
        best = job['best_padded']
        scalar_best = job['best_scalar']
        # A +1 raw B entry needs M additions, with no new multiplications.
        grouped_mults = 2048 * 512 * best['scalar_inner_sum']
        direct_mults = 2048 * 512 * 2048
        jobs.append({
            **job,
            'checks': checks,
            'histogram_mma16_inner_sum': dict(sorted(distribution.items())),
            'histogram_scalar_inner_sum': dict(sorted(scalar_distribution.items())),
            'histogram_chunk_dimension': dict(sorted(chunk_distribution.items())),
            'extra_dense_mma16_work_reduction_vs_zero_b': 1 - best['mma16_inner_sum'] / baseline['mma16_inner_sum'],
            'dense_mma16_work_reduction_vs_direct': 1 - best['mma16_inner_sum'] / 2048,
            'best_scalar_work_reduction_vs_zero_b': 1 - scalar_best['scalar_inner_sum'] / baseline['scalar_inner_sum'],
            'selected_grouped_noise_scalar_multiplications': grouped_mults,
            'raw_b_correction_additions_per_candidate': 2048,
            'direct_scalar_multiplications': direct_mults,
            'csv_file': compressed.name,
            'csv_sha256': sha256(csv_path),
            'csv_gzip_sha256': sha256(compressed),
            'csv_gzip_bytes': compressed.stat().st_size,
            'distinct_committed_b_inputs': len(positions),
            'distinct_b_roots': len(roots),
            'distinct_b_seeds': len(seeds),
        })
    total_checks = [c for job in jobs for c in job['checks']]
    report = {
        'schema': 1,
        'experiment': 'Single-entry B commitment search for cheaper signed-noise graphs',
        'scope': screen['scope'],
        'shape': {'m': 2048, 'n': 512, 'k': 2048, 'r': 128, 'h': 8, 'w': 32},
        'candidate_construction': 'B transpose has one byte +1 at (trial * 1000003) mod 1048576; all other bytes zero. Four fixed synthetic job keys matching prior batch fixtures.',
        'selection': 'Primary: minimum sum(16*ceil(chunk_dimension/16)), then raw dimension sum, then trial. Secondary reverses the first two criteria. No chunks from different commitments are mixed.',
        'timing_definition': 'Fields ending in cpu_seconds are sums of Instant elapsed wall times for a single-thread CPU phase, not process CPU counters. CSV formatting/writing, validation, B preparation and GPU execution are excluded from these phase timers. Total elapsed time includes logging and checks but excludes compilation and this analysis.',
        'excluded_costs': ['Selected B noise materialization and contrast preparation', 'Fresh A commitments, A noise and grouping during mining', 'INT8 coefficient residual corrections', 'Full candidate GPU execution and final hashes', 'Winning proof generation and network acceptance'],
        'raw_b_correction': 'Grouped C increment gains A_prime[:,s] in column j for B[j,s]=1. M additional additions and no additional scalar multiplications per candidate; all subsequent prefixes retain this contribution.',
        'amortization': 'If a selected B actually saves delta_gpu_seconds per fresh A, break-even reuse exceeds search_wall_seconds * cpu_price_per_hour / (delta_gpu_seconds * gpu_price_per_hour), plus other extra costs. delta_gpu_seconds is not measured here. Search loses its applicability when its bound job key changes.',
        'full_commitment_bytes_hashed': sum(j['trials'] for j in jobs) * 512 * 2048,
        'commitment_optimization': 'Every tested B is fully rehashed. Incremental Merkle-tree updates and search GPU acceleration are not evaluated; these timings are not a lower bound on search cost.',
        'total_trials': sum(j['trials'] for j in jobs),
        'raw_files_repository_directory': 'results/seed-search-raw',
        'raw_screen_file': 'screen.json',
        'prefix_entries_checked': sum(c['prefix_entries_checked'] for c in total_checks),
        'tile_states_and_hashes_checked': sum(c['tile_states_and_hashes_checked'] for c in total_checks),
        'correction_negative_controls': sum(c['correction_negative_controls'] for c in total_checks),
        'jobs': jobs,
        'screen_wall_seconds': screen['wall_seconds'],
        'screen_sha256': sha256(screen_path),
        'source_sha256': {str(p.relative_to(source_dir)): sha256(p) for p in [source_dir/'Cargo.toml', source_dir/'Cargo.lock', source_dir/'src/main.rs', source_dir/'src/seed_search.rs', source_dir/'src/grouped_noise.rs']},
        'analysis_script_sha256': sha256(Path(__file__).resolve()),
        'environment': {'platform': platform.platform(), 'machine': platform.machine(), 'rustc': subprocess.check_output(['rustc', '--version'], text=True).strip()},
    }
    output.write_text(json.dumps(report, indent=2) + '\n')
    for j in jobs:
        print(f"job {j['job_id']}: {j['baseline']['mma16_inner_sum']} -> {j['best_padded']['mma16_inner_sum']} MMA16 inner terms ({100*j['extra_dense_mma16_work_reduction_vs_zero_b']:.3f}% extra reduction); search {j['search_cpu_seconds']:.3f}s")
    print(f"Validated {report['total_trials']} trials, {report['prefix_entries_checked']} prefixes, {report['tile_states_and_hashes_checked']} states/hashes, {report['correction_negative_controls']} correction negative controls")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--source-dir', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.directory, args.source_dir, args.output)
