"""
Unified input resolution for the chart analysis pipeline.

Accepts a path that may be:
  - A single image file
  - A single PDF file
  - A directory containing images, PDFs, or a mix of both

Returns a flat list of ResolvedAsset objects where each entry
points to a real raster file ready for cv2.imread().
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'})
PDF_EXTENSION = '.pdf'


class ResolvedAsset(NamedTuple):
    """A single analysis unit — always a real raster file on disk."""
    image_path: Path
    source_document: Optional[str]
    page_index: Optional[int]
    figure_id: Optional[str]


def _make_figure_id(pdf_stem: str, page_num: int, img_index: int) -> str:
    """Generate a deterministic figure identifier for a PDF-extracted chart."""
    return f"{pdf_stem}_p{page_num:03d}_f{img_index:02d}"


def _run_pdf_processor(
    pdf_path: Path,
    output_dir: Path,
    high_res_dpi: int = 300,
    min_chart_width: int = 300,
    min_chart_height: int = 200,
) -> List[dict]:
    """Lazy-import wrapper around process_pdf_charts_optimized.

    Catches ImportError/ModuleNotFoundError when core.pdf_processor (or one of
    its dependencies) is unavailable, logs a clear message, and returns [].
    This is also the test patch target.
    """
    try:
        from core.pdf_processor import process_pdf_charts_optimized
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning(
            "PDF support unavailable (missing dependency while importing core.pdf_processor): %s. "
            "Skipping %s.", exc, pdf_path.name,
        )
        return []

    return process_pdf_charts_optimized(
        pdf_path=pdf_path,
        output_dir=output_dir,
        high_res_dpi=high_res_dpi,
        min_chart_width=min_chart_width,
        min_chart_height=min_chart_height,
    )


def _expand_pdf(
    pdf_path: Path,
    render_dir: Path,
    high_res_dpi: int = 300,
    min_chart_width: int = 300,
    min_chart_height: int = 200,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[ResolvedAsset]:
    """Render all chart images from a PDF into render_dir.

    Returns [] on total failure so the caller's batch can continue.
    """
    if progress_callback:
        progress_callback(f"Extracting charts from {pdf_path.name}...")

    assets: List[ResolvedAsset] = []
    try:
        charts = _run_pdf_processor(
            pdf_path=pdf_path,
            output_dir=render_dir,
            high_res_dpi=high_res_dpi,
            min_chart_width=min_chart_width,
            min_chart_height=min_chart_height,
        )
    except Exception as exc:
        logger.error("PDF expansion failed for %s: %s", pdf_path, exc)
        return []

    for chart in charts:
        high_res_path = chart.get('high_res_path')
        if high_res_path is None or not Path(high_res_path).exists():
            logger.warning(
                "Skipping chart with missing high_res_path from %s", pdf_path.name,
            )
            continue

        page_num = chart.get('page_num', 0)
        img_index = chart.get('image_index', 0)
        figure_id = _make_figure_id(pdf_path.stem, page_num, img_index)

        assets.append(ResolvedAsset(
            image_path=Path(high_res_path),
            source_document=str(pdf_path),
            page_index=page_num,
            figure_id=figure_id,
        ))

    logger.info("PDF %s expanded to %d chart(s)", pdf_path.name, len(assets))
    return assets


def _resolve_single_file(
    file_path: Path,
    render_dir: Path,
    input_type: str,
    high_res_dpi: int,
    min_chart_width: int,
    min_chart_height: int,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[ResolvedAsset]:
    suffix = file_path.suffix.lower()

    if input_type in ('auto', 'image') and suffix in IMAGE_EXTENSIONS:
        return [ResolvedAsset(
            image_path=file_path,
            source_document=None,
            page_index=None,
            figure_id=None,
        )]

    if input_type in ('auto', 'pdf') and suffix == PDF_EXTENSION:
        try:
            render_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.error("Failed to create PDF render directory %s: %s", render_dir, exc)
            return []
        return _expand_pdf(
            file_path, render_dir, high_res_dpi,
            min_chart_width, min_chart_height, progress_callback,
        )

    logger.warning(
        "File %s skipped (input_type=%s, suffix=%s)",
        file_path.name, input_type, suffix,
    )
    return []


def _resolve_directory(
    dir_path: Path,
    render_dir: Path,
    input_type: str,
    high_res_dpi: int,
    min_chart_width: int,
    min_chart_height: int,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[ResolvedAsset]:
    assets: List[ResolvedAsset] = []

    all_files = sorted(p for p in dir_path.iterdir() if p.is_file())
    image_files = [p for p in all_files if p.suffix.lower() in IMAGE_EXTENSIONS]
    pdf_files = [p for p in all_files if p.suffix.lower() == PDF_EXTENSION]

    # Native images
    if input_type in ('auto', 'image'):
        for img in image_files:
            assets.append(ResolvedAsset(
                image_path=img,
                source_document=None,
                page_index=None,
                figure_id=None,
            ))

    # PDFs
    if input_type in ('auto', 'pdf'):
        try:
            render_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.error("Failed to create PDF render directory %s: %s", render_dir, exc)
            pdf_files = []  # Skip PDF expansion if directory creation fails

        for pdf in pdf_files:
            pdf_assets = _expand_pdf(
                pdf, render_dir, high_res_dpi,
                min_chart_width, min_chart_height, progress_callback,
            )
            assets.extend(pdf_assets)

    if not assets:
        logger.warning(
            "No processable files found in %s (input_type=%s). "
            "Found %d image(s), %d PDF(s).",
            dir_path, input_type, len(image_files), len(pdf_files),
        )

    return assets


def resolve_input_assets(
    input_path: Path,
    render_dir: Path,
    input_type: str = 'auto',
    high_res_dpi: int = 300,
    min_chart_width: int = 300,
    min_chart_height: int = 200,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[ResolvedAsset]:
    """Resolve an input path into a flat, ordered list of ResolvedAsset objects.

    Parameters
    ----------
    input_path:
        May be a single file (image or PDF) or a directory.
    render_dir:
        Directory where PDF-rendered PNGs are saved.
    input_type:
        'auto'  — detect by file extension (default)
        'image' — treat every file as an image; skip PDFs
        'pdf'   — treat every file as a PDF; skip native images
    high_res_dpi, min_chart_width, min_chart_height:
        Passed through to process_pdf_charts_optimized for PDF rendering.
    progress_callback:
        Optional callable receiving status strings (for GUI progress display).

    Returns
    -------
    Sorted list of ResolvedAsset. Empty list if nothing usable was found.
    """
    input_path = Path(input_path)
    render_dir = Path(render_dir)

    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        return []

    if input_path.is_file():
        return _resolve_single_file(
            input_path, render_dir, input_type,
            high_res_dpi, min_chart_width, min_chart_height,
            progress_callback,
        )

    if input_path.is_dir():
        return _resolve_directory(
            input_path, render_dir, input_type,
            high_res_dpi, min_chart_width, min_chart_height,
            progress_callback,
        )

    logger.error("Input path is neither a file nor a directory: %s", input_path)
    return []


def asset_provenance_dict(asset: ResolvedAsset) -> Optional[Dict[str, Any]]:
    """Return a JSON-safe provenance dict, or None for native images.

    Only includes str/int fields — no Path objects — so the result
    is safe for json.dump without custom serializers.
    """
    if asset.source_document is None:
        return None
    return {
        'source_document': asset.source_document,
        'page_index': asset.page_index,
        'figure_id': asset.figure_id,
    }
