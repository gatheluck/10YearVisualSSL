# Anonymous supplementary source archive

Status, support and validation statements describe this portable package at the
date recorded; see [scope and terminology](SUBMISSION_SCOPE.md)
for the distinction from original experimental implementations and results.

`bin/submission-archive.py` audits a committed Git tree and creates a
deterministic ZIP only when the audit passes. It does not modify the checkout,
fetch dependencies, upload anything, or establish scientific reproducibility.
Python's standard library and Git are required.
Run from the checkout root, or specify that root with `--repo`. A subdirectory
is rejected so it cannot weaken the private-output boundary.

The policy, report and ZIP must be kept outside the checkout. The report contains
private paths and commit identifiers: **never attach it to a submission**.
The ZIP contains selected files under `code/`, with fixed timestamps and no Git
history or generated provenance manifest. Executable file permissions survive.
Git metadata, including `.gitmodules`, is omitted and recorded in the report.

## Prepare a private policy early

Copy [the policy example](../configs/submission-policy.example.json) to a private
directory outside Git. For example, from the repository root:

```bash
mkdir -p "$HOME/.local/state/submission-archive"
cp configs/submission-policy.example.json "$HOME/.local/state/submission-archive/policy.json"
```

The example selects **only README.md**, plus automatically retained license
notices. It is a schema example, not a complete experiment export or a usable
anonymization policy. Replace the placeholder forbidden terms with every author's
names, aliases, account names, affiliations and identifying project names.
Add the source, configurations, dependency files, tests and instructions needed
for the submitted experiments to `include`. Review everything excluded; a ZIP
can pass this audit while missing code needed to reproduce an experiment.

`include` and `exclude` contain literal relative POSIX file/directory paths, not
globs. Missing includes and unused edit/approval entries block generation.
Unknown fields and unsafe paths are errors. `max_bytes` limits both original
selected bytes and final uncompressed bytes; it is not a conference upload limit.
Reaching the input limit stops reading and blocks the incomplete audit.

Only **committed** bytes are read. Commit intended changes before building, and
use a full commit hash with `--ref` for the final artifact. Dirty and untracked
files are ignored, including dirty submodules. Selected submodules must already
be initialized and have the parent-recorded commit locally available. Their
pinned source contents are materialized recursively; their remote URL and Git
metadata are not packaged. The tool never downloads missing objects.

When upstream source is bundled, the ZIP also contains the generated
`upstream_sources.json`: only a schema version and relative directories that
actually contain bundled files. It contains no account, URL or Git revision.
This reserved file cannot be supplied or edited by a policy. It lets the shared
file/dependency scanner recognize bundled packages without `.gitmodules`;
unknown external imports are still rejected. The reader validates the schema,
existing directories, duplicates and unsafe/symlink paths. Existing `.gitmodules`
remains authoritative in a checkout. The private report marks generated content
explicitly, with no fictitious source-blob hash.

## Audit, resolve findings, then build

Run without `--out` to audit without creating an archive:

```bash
python3 bin/submission-archive.py --ref HEAD --policy "$HOME/.local/state/submission-archive/policy.json" --report "$HOME/.local/state/submission-archive/audit.json"
```

Read the private JSON report's `findings`, `excluded`, `files` and `submodules`.
Each finding names a path and rule, without copying the matched contents.
The report and new ZIP use owner-only permissions. Exit codes are 0 for success,
1 for blocking findings, and 2 for invalid policy/source/output or an I/O error.
Invalid inputs can prevent creation of a report; do not treat an old report as
evidence of the latest run. The CLI deliberately does not print private inputs.

After resolving findings, create the archive (replace HEAD with the reviewed
commit for final delivery):

```bash
python3 bin/submission-archive.py --ref HEAD --policy "$HOME/.local/state/submission-archive/policy.json" --report "$HOME/.local/state/submission-archive/audit.json" --out "$HOME/.local/state/submission-archive/code.zip"
```

Existing ZIP files are never overwritten. Choose a fresh anonymous filename for
another run. The final report records `status: created` and the archive SHA256.
Identical source, policy and Python/zlib environment produce identical ZIP bytes.
Unpack and manually inspect the ZIP, then run the relevant experiment smoke tests
from the unpacked copy before attaching **only the ZIP** to the submission.

## Resolve findings without hiding them

The detector scans file names and text for the forbidden terms, case-insensitively
with Unicode, percent/entity and invisible-format normalization. It also detects
email addresses, local account/cluster paths, common private-key/token markers,
web/Git links and embedded base64 media. Complete scheduler scratch templates using only a
job ID are recognized as generic infrastructure paths; concrete account paths,
unknown expressions and trailing private directories remain blocked. These conservative rules can flag public
upstream references. They are not exhaustive secret or identity detection.

For a reviewed text-only submission variant, add a full-file replacement:

```json
{"README.md": {"sha256": "<64 lowercase hex digits of original committed bytes>", "text": "<complete anonymous text>", "reason": "<review rationale>"}}
```

Place this object in `replacements`. The original stays unchanged. A changed
source hash blocks the old replacement; replacement contents are scanned again.
This is suitable for an anonymous README, not blanket string deletion from code.

License/notice files are retained in each visited repository even when omitted
from `include`. Explicitly excluding them blocks generation. License files and
source files with recognized copyright/SPDX headers cannot be replaced through
this tool. A narrowly scoped exception is available for the project's own root `LICENSE`,
when the rights holder authorizes an anonymous review copy. Add the optional
`first_party_license` policy field:

```json
{"sha256": "<original root LICENSE SHA256>", "copyright_line": "Copyright (c) 2026 <rights holder>", "authorized": true, "reason": "<record of the rights-holder instruction>"}
```

The tool requires exactly one matching complete copyright line and the original
file hash. It changes only that line's holder to `Anonymous authors`, preserving
the year, all other bytes and the original checkout. No path selector is accepted:
this cannot authorize edits to vendor licenses or source copyright headers.
All remaining content is scanned normally; the private report records the
original and packaged hashes. General license replacement/exclusion stays blocked.
The authorization is the caller's attestation, not a legal determination by the
tool. Do not use it for rights owned by third parties or treat a passed audit as
licensing permission.

A manually inspected public reference or binary can receive a narrowly scoped
approval in `approvals`:

```json
{"vendor/source.py": {"sha256": "<64 lowercase hex digits of final packaged bytes>", "rules": ["external-link"], "reason": "<why this upstream reference is safe>"}}
```

Only `external-link`, `email`, `binary` and `opaque` can be approved. A changed
hash blocks the approval. Identifiers, local paths and secret markers cannot be
waived. A binary approval still scans visible decoded text. Review binary
metadata and visible content manually; notebooks require `opaque` approval even
when their JSON contains no detectable identifier. Symlinks, Git LFS pointers,
case/Unicode path collisions and nonportable archive names block generation.

## Boundaries and validation

This is a fail-closed packaging aid for the configured rules, **not a guarantee
of anonymity**. Unknown author aliases, obfuscated/encoded text, images, metadata,
indirect identifying links and prose can evade automated checks. Hash approvals
record human review, not proof of safety. Source selection and copyright-header
recognition also require manual review. Passing this audit says nothing about
whether all paper experiments, weights or datasets are included or reproducible.

Before rebuilding at a newer commit, compare the selected files against current
runtime imports and document links. Exact-file include lists can omit newly
added modules even when old selections still exist. Re-review changed content
before refreshing hash-bound replacements or approvals; do not merely update
hashes. Check the actual unpacked archive, not only the source checkout.
The [supplied companion protocols](submission_protocols/README.md) distinguish
received specifications from pending inputs and implemented components. Include
that directory when packaging these specifications and resolve its pending
companion inputs before describing the bundle as complete.

Behavioral tests use real temporary Git repositories and submodules, exercise
the CLI, unzip and execute packaged fixture code, check repeatability and
original-file preservation, and reject unsafe inputs. Mutation tests verify
that removing rejection paths is detected. These fixtures do not certify the
current project's complete submission archive.

```bash
python3 -m unittest tests.test_submission_archive -v
python3 bin/mutate.py --spec mutations/submission-archive.json
```

## Versioned reviewer README

The [reviewer README template](templates/REVIEWER_README.md) is intended to be
installed as `code/README.md`, not read with template-relative links. Its links
resolve from the export root. Use its exact UTF-8 text as the private policy's
`replacements["README.md"].text`, with the SHA-256 of the selected commit's
original README. The exporter does not select this template automatically.
Include the linked terminology guide, protocol documents and JSON registries
in the allowlist. Review any hash-bound approvals after source changes and
inspect the final ZIP; a checked template does not certify a future archive.

Keep the three Extend registries in Git: they are sanitized experiment
specifications. Keep `upstream_sources.json` generated by the exporter: it
describes the selected bundled dependencies, not a separately maintained input.
Initial protocol reconstructions and later Unified LP/AP/FT specifications
must remain separately indexed. The 2026-09-26 author correction establishes
COCO as detection in the initial companions; it does not retroactively verify
every supplied detector setting against completed run records.
