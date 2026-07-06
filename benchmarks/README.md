# Searchlight Benchmarks

`run_pipeline_benchmark.py` calls `SearchlightPipelineService` directly and records stage timings, crop/detection counts, retained crop boxes, detection signatures, visual output hashes, serialized response bytes, and Python/CUDA peak memory signals.

## Usage

Use committed code with external fixture images:

```powershell
$env:SEARCHLIGHT_BENCHMARK_FIXTURE_DIR="E:\path\to\fixtures"
python benchmarks\run_pipeline_benchmark.py --profile full --runs 1 --warm-runs 1
```

Or pass individual images:

```powershell
python benchmarks\run_pipeline_benchmark.py --image E:\path\frame.png --profile display --runs 3
```

By default, JSON output is written under the system temp directory, outside the repository. To write local throwaway results, use `--output benchmarks\results\run.json`; that directory is ignored. To intentionally keep a curated baseline summary, use `--baseline-name <name>`, review the JSON, and commit it from `benchmarks/baselines/`.

## Comparing Runs

After collecting a baseline and candidate JSON, use:

```powershell
python benchmarks\compare_pipeline_benchmarks.py baseline.json candidate.json
```

The comparison checks matching fixture records, output keys, crop image counts, retained crop boxes, detection classes/confidence/global boxes, visual output hashes, payload bytes, the target stage improvement threshold, and total latency regression. Use the tolerance flags only when fixture evidence supports a deliberate quality delta.
