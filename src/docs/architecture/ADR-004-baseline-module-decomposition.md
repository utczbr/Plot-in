# ADR-004: Baseline Module Decomposition with Compatibility Facade

**Status:** Accepted  
**Date:** 2026-02-08  
**Deciders:** Architecture Team  
**Technical Story:** Baseline decomposition and contract stabilization

## Context and Problem Statement

Baseline detection logic had accumulated in a monolithic module (`src/core/baseline_detection.py`), making it difficult to test, review, and safely evolve. Runtime callers depended on the old import path, so migration had to preserve compatibility.

## Decision Drivers

* **Maintainability:** Reduce blast radius of baseline changes.
* **Testability:** Enable targeted unit tests by responsibility.
* **Compatibility:** Preserve existing imports during migration.
* **Risk Control:** Keep behavior parity while refactoring.

## Decision Outcome

**Chosen Approach:** Decompose baseline internals into a canonical package while keeping `core.baseline_detection` as a compatibility facade.

### Canonical Modules

* `src/core/baseline/detector.py` - orchestrates baseline detection flow.
* `src/core/baseline/policy.py` - chart policy and axis-id mapping.
* `src/core/baseline/geometry.py` - geometry and stacked-end helpers.
* `src/core/baseline/zero_crossing.py` - calibration zero-crossing + interpolation fallback.
* `src/core/baseline/scatter.py` - scatter-specific baseline heuristics.
* `src/core/baseline/stats.py` - reusable statistical primitives.
* `src/core/baseline/types.py` - `BaselineLine`, `BaselineResult`, `DetectorConfig`.

### Compatibility Strategy

* Keep `src/core/baseline_detection.py` as stable entrypoint.
* Re-export historical symbols used by existing handlers and services.
* Avoid runtime contract changes (`ModularBaselineDetector.detect(...) -> BaselineResult`).

## Consequences

**Positive:**
* Smaller, reviewable modules.
* Better focused tests for baseline behavior.
* Safer future refactors and policy changes.

**Negative:**
* Temporary indirection via facade.
* Short-term duplication in docs/import pathways until complete cutover.

## Validation

* Characterization tests lock expected behavior (horizontal zero-crossing, no-elements, scatter dual baseline, fallback interpolation).
* Dual-axis latent path no longer references an undefined method.
* Existing caller imports from `core.baseline_detection` remain valid.
