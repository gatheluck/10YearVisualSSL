# Downstream final-run accounting

See [scope and terminology](SUBMISSION_SCOPE.md) for the distinction between
original experiments, portable components and reproduced results.

`python -m downstream.accounting` audits existing LP/AP/FT outputs without
launching or modifying experiments. It supplements the method-level
`bin/aggregate-seeds.py`; the two contracts describe different run types.

Create a cell plan with explicitly selected attempts:

```json
{
  "expected_seeds": [0, 1, 2],
  "runs": [
    {
      "seed": 0,
      "config": "/path/to/seed0/config.json",
      "out": "/path/to/seed0/output",
      "exit_status": 0
    }
  ]
}
```

Run `python -m downstream.accounting --config /path/to/cell.json --out /path/to/new-report.json`.
An existing report is refused. Exit zero means the audit was written, not that
the experiment is complete: inspect `complete`, `missing_seeds` and
`invalid_runs`. Missing and technically invalid repeats are separate; neither
becomes a measured zero. Valid zero/low scores remain in the mean. Every valid
per-seed score is retained, with actual n, mean and sample standard deviation
(ddof=1). Standard deviation is null for n=1, rather than an invented zero.

The audit checks the existing downstream artifact/config hashes, recorded exit
status, seed/task/backbone/adaptation identity, the final scheduled epoch and
agreement between final results and raw metrics. Different recipes, checkpoints,
dataset locations or metric sets must be separate cells. Duplicate seeds are
refused; select the intended attempt explicitly rather than selecting by score.
Configuration comparison excludes only seed. Paths are not normalized or treated
as equivalent datasets/checkpoints without evidence.

Component runs remain noncanonical and nonrecordable. An aggregate never upgrades
their status. This tool verifies stored output consistency, not the authenticity
of a claimed execution, immutable checkpoint contents, or correspondence with a
paper table. Historical experimental result schemas need explicit conversion
and run mapping before this checker can accept them. The tool does not infer
missing experiments or perform frontier/Extended evaluations.
