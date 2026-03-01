# Contributing Guide

Last verified: **February 23, 2026**.

## Goals
Contributions should preserve runtime stability, protocol output integrity, and documentation accuracy.

## Core Rules
1. Keep behavior claims evidence-based.
2. Preserve backward compatibility unless the PR explicitly changes contracts.
3. Add or update tests for any behavior or contract change.
4. Keep docs synchronized with code in the same PR.

## Coding Standards
- Python style: PEP 8
- Formatting: `black`
- Import ordering: `isort`
- Type hints: required on new/changed public functions
- Logging: use `logging`; do not add ad-hoc debug flags
- Remove dead/commented-out code before merge

Recommended local checks:
```bash
black src tests
isort src tests
```

## Runtime Contract Areas (High-Risk)
If your PR touches any of these, include explicit contract notes in the PR description:
- `src/pipelines/types.py` (`PipelineResult`)
- `src/core/protocol_row_builder.py` (protocol row schema)
- `src/core/export_manager.py` (protocol CSV columns)
- `src/validation/run_protocol_validation.py` (gating metrics)

## Required Test Matrix By Change Type

### Input/PDF/ingestion changes
```bash
python3 -m pytest tests/core_tests/test_input_resolver.py
```

### Chart routing/handler support changes
```bash
python3 -m pytest tests/core_tests/test_orchestrator_registry.py
python3 -m pytest tests/handlers_tests/test_area_handler.py
```

### Protocol row or CSV changes
```bash
python3 -m pytest tests/core_tests/test_protocol_row_builder.py
python3 -m pytest tests/core_tests/test_export_manager_protocol.py
```

### Validation metric or gate changes
```bash
python3 -m pytest tests/evaluation_tests/test_protocol_validation.py
python3 -m pytest tests/evaluation_tests/test_accuracy_comparator_metrics.py
```

## CI Awareness
Current workflows:
- `.github/workflows/evaluation-tests.yml`
- `.github/workflows/installer-build.yml`

If your change introduces a new mandatory quality gate, update workflow coverage in the same PR.

## Documentation Sync Requirements
When changing runtime behavior, update docs in the same branch:
- `Protocol_gap.md`
- `src/Critic.md`
- `src/README.md`
- `src/CONTRIBUTING.md` (if process changed)

Use `src/docs/DOC_ACCURACY_CHECKLIST.md` and ensure major claims cite current code paths and tests.
Use `src/docs/README.md` to keep active-doc vs archive links correct.

## Pull Request Checklist
- [ ] Code formatted and imports sorted
- [ ] Relevant tests pass locally
- [ ] Runtime contract changes documented
- [ ] Protocol output implications documented
- [ ] Docs updated with evidence references
- [ ] Backward compatibility impact stated

## Branch And PR Workflow
1. Create feature branch from `main`.
2. Implement change + tests.
3. Run relevant local test matrix.
4. Update affected docs.
5. Open PR with:
   - problem statement
   - behavior delta
   - test evidence
   - contract impact

## Guidance For Protocol-Output Changes
Any change affecting protocol rows, CSV schema, or validation metrics must include:
1. Before/after schema notes.
2. Fixture updates (if required).
3. Evidence from protocol row and export tests.
4. Validation harness impact summary.
