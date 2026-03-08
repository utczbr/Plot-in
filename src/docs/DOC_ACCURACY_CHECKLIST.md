# Documentation Accuracy Checklist

Last updated: **February 23, 2026**.

## Purpose
Manual governance checklist to ensure active documentation stays aligned with current code and tests.

## Active Docs Under Governance
- `src/docs/Context.md`
- `src/docs/TESTING_WITH_ANALYSIS.md`
- `src/docs/DOC_ACCURACY_CHECKLIST.md`
- `src/docs/RD_NOTES.md`
- `src/docs/README.md`
- `Protocol_gap.md`
- `src/Critic.md`
- `src/README.md`
- `src/CONTRIBUTING.md`

## Claim Evidence Rule
For each major claim, provide:
1. At least one current code reference
2. At least one test reference (when test coverage exists)
3. Verification date
4. Status label: `Correct`, `Outdated`, `Partially correct`, or `Unverifiable`

## Minimum Detail Rule (Pipeline Sections)
Each major pipeline step section must include:
1. Inputs
2. Core stage behavior
3. Outputs/contract fields
4. Fallback behavior
5. Known failure modes
6. Test references
7. Protocol implications

## Archive Governance Rule
A doc may be archived only if all are true:
1. It is proposal/legacy content and not runtime truth.
2. Any still-useful content is condensed into an active doc first.
3. Archive entry is added in `src/docs/archive/README.md` with:
   - original path
   - archive date
   - reason
   - replacement doc
4. No active non-archive doc links directly to archived content.

## Status Ledger Template

| Doc | Claim Summary | Status | Code Evidence | Test Evidence | Verified On | Reviewer |
|---|---|---|---|---|---|---|
| src/docs/Context.md | 8-type support and runtime stages | Correct | `src/pipelines/chart_pipeline.py`, `src/ChartAnalysisOrchestrator.py` | `tests/core_tests/test_orchestrator_registry.py`, `tests/pipelines_tests/test_chart_pipeline.py` | 2026-02-23 | pending |
| src/docs/TESTING_WITH_ANALYSIS.md | CLI/protocol/validation workflows | Correct | `src/analysis.py`, `src/validation/run_protocol_validation.py` | `tests/evaluation_tests/test_protocol_validation.py` | 2026-02-23 | pending |
| src/docs/RD_NOTES.md | Marked non-runtime exploratory ideas | Correct | `src/docs/RD_NOTES.md` | N/A | 2026-02-23 | pending |
| src/docs/README.md | Active-vs-archive doc map | Correct | `src/docs/README.md`, `src/docs/archive/README.md` | N/A | 2026-02-23 | pending |

## Red-Flag Triggers (Rewrite/Remove Required)
1. Claim references code paths that no longer exist.
2. Claim contradicts current chart registry or handler mapping.
3. Claim says a feature is missing when runtime/tests show it is implemented.
4. Claim has no verification date.
5. Active docs link to archived docs as if they were runtime truth.

## Merge Gate (Manual)
- [ ] All new major claims include code references.
- [ ] Test references added where test coverage exists.
- [ ] Outdated/unverifiable claims removed or rewritten.
- [ ] Verification dates refreshed in edited docs.
- [ ] Archived docs recorded in `src/docs/archive/README.md`.
- [ ] No active doc contains links to archived docs.
