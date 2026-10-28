# Manuscript storage cleanup — October 28, 2026

## Result

The local cleanup is complete and independently verified. It reclaimed
199,423,830,886 bytes (185.73 GiB) by removing verified redundant native simulator
files after establishing their lossless archive representation. The measured C:
free-space increase was 197,464,932,352 bytes (183.90 GiB), from 70.39 GiB to
254.30 GiB; concurrent system/OneDrive activity accounts for the difference
between file-accounting savings and the disk observation.

The user explicitly requested preserving everything needed for both the original
manuscript and its revision, including results not yet integrated into the text.
No performance-based run selection was performed. Scientific results with adverse
or mixed outcomes, failed-attempt records, and numerical provenance remain.

Permanent cloud cleanup is a separate pending step. The browser reported
0.00 GB free of 1024.00 GB. A concrete selection of 211 deleted native files
(19.83 GiB) matches intact lossless archives by original path, saved full file
SHA-256, and displayed cloud size. The broader recycle bin contains unrelated
files and must not be emptied indiscriminately. No permanent cloud deletion had
occurred at completion of the local verification.

## What is retained

- All 46 frozen original manuscript files, with matching SHA-256 values.
- Every source bundle in both original manuscript run manifests: 258 manifest
  rows across the canonical and repeated-run analyses. Online-only sources were
  checked by metadata and excluded from cleanup without downloading them.
- All 100 completed revision Scenario 2 training runs, both SG and plain,
  all prescribed families, plants and seeds.
- All 3,840 primary E1 paired traces, decision records, portable snapshots and
  available selected native snapshots; native model files are either ready at
  their original paths or recoverable from verified archive mappings.
- All 12 E2 controller/reference trajectories, E3 raw probes and the final
  576-row analysis, policy exports, identification inputs, analysis scripts,
  figures, final assessment packages, and source/configuration records.
- Both final complete Python/native checkpoints for each of the five column E1
  configurations: steps 120000 and 119600, ten complete sets in total. These stay
  unpacked in their original directories. Final-generation local backup copies
  and any unclassified unique native backups were excluded.
- Every pre-existing file outside the exact native removal list: 57,320 files
  passed the independent final size/timestamp check. No raw numerical result,
  policy, Python checkpoint, manuscript, original model or source file was a
  removal target. The pre-existing dirty polymer runner remains untouched.

## Why the runs grew so large

The initial inventory found 335.69 GiB of logical files in the code repository
and 165.61 GiB in the local recovery archive. Within the repository, `.dynf`
files accounted for 162.99 GiB and `.pkl` files for 111.22 GiB. A late full Aspen
document could be about 0.84 GiB. Hundreds of matched-decision snapshots and
recovery/load copies multiplied those documents. Full learner checkpoints also
retain replay buffers and optimizer/controller state; a polymer checkpoint
exceeded 1 GiB. Their size is unrelated to the size of the manuscript figures.

The earlier 910-folder historical archive had already been replaced with
verified ZIPs and its raw duplicates removed on October 23. This cleanup did
not claim those earlier savings again. The later experiment/recovery outputs
caused the present accumulation. The existing background retainer's saved
status was stale after the machine restarted; a fresh full-process inspection
found no scientific worker or Aspen Dynamics simulation running. Nothing was
stopped or restarted by this cleanup.

## Exact cleanup and verification

1. Inventory metadata across both repositories and the explicitly named local
   archive, without following junctions or hydrating online-only files.
2. Verify 4,271 protected sources: 4,033 local files by fresh SHA-256 and 238
   online-only files by metadata. Verify the ten final Python/native pairs and
   their saved linkage/hashes. OneDrive had updated directory timestamps, so
   the live-worker guard conservatively deferred; its implementation was left
   unchanged. The separate closed-run audit checked immutable content and
   before/after file generations instead.
3. Pass six guard/restore tests using disposable fixtures, then perform and
   hash-check an actual 906,005,432-byte staged native-file restoration.
4. Compact 536 historical `.dynf` files, totaling 150.69 GiB. Most already had
   byte-identical ZIP members. Only 116 new unique ZIPs (3.64 GiB) were needed.
   New archives are under `archive/manuscript_native_states_20261028/` and are
   excluded from Git by the existing `/archive/` ignore rule.
5. Remove 81 additional raw native backup duplicates (38.68 GiB) from
   `C:\Users\hamediaa\RL_MPC_local_archive`, only after a read-only exact plan
   and full source/archive-member verification. No existing ZIP was removed.
6. Independently rehash the 4,033 retained local sources, all final checkpoint
   components, and all 294 ZIP files supporting this cleanup. Confirm every
   deletion against the durable receipts and verify all other inventoried
   files' metadata. `final_verification.json` reports zero errors.

Before each unlink, the helper checked the resolved allowed root, excluded
protected/final paths and online-only files, refused changed generations and
open native files, and verified the original bytes against a fully decompressed
ZIP member. It flushed a restore receipt before removal and recorded the result
afterwards. No directory was recursively deleted.

## Restoration

`native_compaction_receipts.jsonl` maps each compacted original path to a ZIP,
member, original SHA-256 and ZIP SHA-256. `native_backup_duplicate_receipts.jsonl`
contains the corresponding mappings for the local archive duplicates. Keep
these files together with the existing archive receipts and archives.

For a repository native file, run `compact_native.py restore --source
'ORIGINAL_ABSOLUTE_PATH' --destination 'NEW_STAGING_FILE_PATH'` using the existing
`rl-assisted-env` interpreter. The destination parent must exist and the file
must not exist. The helper checks the ZIP and every restored byte and refuses
overwrites. Restore a historical dependency before attempting to open/replay it.
No restoration is required for normal figure/table analysis or the ten unpacked
final checkpoint sets.

For a removed local backup, find its `preservation.archive`, `member`, `sha256`
and byte count in the backup receipt, extract that single member to a new
staging path, and compare its SHA-256 before use. Existing historical archives
retain their original restoration utilities.

## Evidence files and scope

Primary evidence is `inventory_summary.json`, `protected_evidence_before.json`,
`native_compaction_plan.json`, both removal receipt logs, the backup plan/result,
`pilot_restore_verification.json` and `final_verification.json`. The complete
file-level inventories are retained locally alongside this report. The
`cloud_review/` manifests record exact cloud candidates and verified archive
representations; other recycle-bin items remain outside scope.

Files changed are the cleanup helpers and their audit records in this directory,
the 617 individually verified native-file removals, 116 retained ZIP additions,
and paper-workflow records documenting preservation. Scientific source,
manuscript TeX/PDF, original numeric data and the unrelated dirty runner were
not edited. No new experiment, scientific claim, cloud upload command or
recurring automation was introduced.
