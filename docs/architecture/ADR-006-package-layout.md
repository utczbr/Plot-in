# ADR-006: Packaging Architecture, Entry Points, and Import Structure

## Context
`Plot-in` is packaged using a standard PEP 517 build backend ([pyproject.toml](file:///home/stuart/Documentos/OCR/LYAA-fine-tuning/pyproject.toml)).

The codebase layout features subpackages directly under `src/` (`core`, `services`, `visual`, `handlers`, `extractors`, `strategies`, etc.) as well as a top-level `shared/` directory residing at the repository root outside `src/`.

## Decisions

1. **Package Directory & Import Mapping:**
   To maintain backward compatibility without rewriting hundreds of internal module imports, `setuptools` is configured with:
   ```toml
   [tool.setuptools.package-dir]
   "" = "src"
   "shared" = "shared"
   ```
   This exposes `shared` alongside `src/` subpackages in standard wheel builds while keeping module imports unchanged (`from shared.state_root import ...`).

2. **Console Script Entry Points:**
   Three executable console scripts are defined in `pyproject.toml`:
   - `plotin = "analysis:main"` (CLI batch processing entry point)
   - `plotin-gui = "main_modern:main"` (Desktop PyQt6 application entry point)
   - `plotin-fetch-models = "scripts.fetch_models:main"` (Standalone model downloader)

   The GUI entry point `main_modern:main` wraps full application initialization (logging setup, profile application, exception hook, and service container construction).

## Trade-offs & Accepted Risks
Top-level module names (`core`, `services`, `handlers`, `extractors`, `utils`) are non-namespaced. If installed into a shared global Python environment alongside third-party libraries using identical top-level names, collisions could occur. This trade-off is accepted for internal app deployment and virtual environment isolation.
