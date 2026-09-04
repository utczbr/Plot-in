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
    image_buffer: Optional[Any] = None
    confidence_info: Optional[Dict[str, Union[float, bool, str]]] = None


def _make_figure_id(pdf_stem: str, page_num: int, img_index: int) -> str:
    """Generate a deterministic figure identifier for a PDF-extracted chart."""
    return f"{pdf_stem}_p{page_num:03d}_f{img_index:02d}"


def _run_pdf_processor(
    pdf_path: Path,
    output_dir: Path,
    high_res_dpi: int = 200,
    min_chart_width: int = 300,
    min_chart_height: int = 200,
    cancel_event: Optional[Any] = None,
    model_manager: Optional[Any] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kwargs,
) -> List[dict]:
    """Helper wrapping process_pdf_charts_optimized.

    If process_pdf_charts_optimized (or fitz/numpy/cv2 imported by
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
        cancel_event=cancel_event,
        model_manager=model_manager,
        progress_callback=progress_callback,
        **kwargs,
    )


def _expand_pdf(
    pdf_path: Path,
    render_dir: Path,
    high_res_dpi: int = 200,
    min_chart_width: int = 300,
    min_chart_height: int = 200,
    progress_callback: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[Any] = None,
    failure_callback: Optional[Callable[[Path, str], None]] = None,
    model_manager: Optional[Any] = None,
    force_reextract: bool = False,
    check_cache_only: bool = False,
) -> List[ResolvedAsset]:
    """Render all chart images from a PDF into render_dir.

    Returns [] on total failure so the caller's batch can continue.
    """
    if progress_callback:
        progress_callback(f"Extracting charts from {pdf_path.name}...")

    charts = None
    if not force_reextract and render_dir.exists():
        metadata_file = render_dir / f"{pdf_path.stem}_processing_metadata.json"
        if metadata_file.exists() and pdf_path.exists():
            try:
                if metadata_file.stat().st_mtime >= pdf_path.stat().st_mtime:
                    import json
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                    cached_charts = meta_data.get("charts", [])
                    if cached_charts:
                        all_exist = True
                        for c in cached_charts:
                            hr_p = Path(c.get("high_res_file", ""))
                            if not hr_p.exists():
                                all_exist = False
                                break
                        if all_exist:
                            charts = [{
                                'page_num': c.get('page_num', 1),
                                'image_index': idx + 1,
                                'high_res_path': c.get('high_res_file'),
                                'dimensions': c.get('dimensions'),
                                'confidence_info': c.get('confidence_info'),
                            } for idx, c in enumerate(cached_charts)]
                            logger.info("Reusing %d pre-rendered chart(s) for %s from %s", len(charts), pdf_path.name, metadata_file.name)
            except Exception as exc:
                logger.debug("Failed loading cached PDF metadata for %s: %s", pdf_path.name, exc)
                charts = None

    if charts is None:
        if check_cache_only:
            return []
        try:
            charts = _run_pdf_processor(
                pdf_path=pdf_path,
                output_dir=render_dir,
                high_res_dpi=high_res_dpi,
                min_chart_width=min_chart_width,
                min_chart_height=min_chart_height,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
                model_manager=model_manager,
            )
        except Exception as exc:
            logger.error("PDF processing failed for %s: %s", pdf_path, exc, exc_info=True)
            if failure_callback is not None:
                try:
                    failure_callback(pdf_path, str(exc))
                except Exception:
                    pass
            return []

    if not charts:
        logger.info("No charts extracted from PDF %s", pdf_path.name)
        return []

    assets: List[ResolvedAsset] = []
    for chart in charts:
        high_res_path = chart.get('high_res_path')
        if not high_res_path or not Path(high_res_path).exists():
            continue

        page_num = chart.get('page_num', 0)
        img_index = chart.get('image_index', 0)
        figure_id = _make_figure_id(pdf_path.stem, page_num, img_index)

        assets.append(ResolvedAsset(
            image_path=Path(high_res_path),
            source_document=str(pdf_path),
            page_index=page_num,
            figure_id=figure_id,
            image_buffer=chart.get('image_buffer'),
            confidence_info=chart.get('confidence_info'),
        ))

    # Upfront classification on PDF-extracted assets if model_manager is available and loaded
    if model_manager is not None and getattr(model_manager, '_models', None) and assets:
        needs_classification = [idx for idx, a in enumerate(assets) if a.confidence_info is None]
        if needs_classification:
            try:
                from core.ensemble_classifier import WeightedChartClassifier
                from services.confidence_extractor import LOW_CONFIDENCE_THRESHOLD
                import cv2

                classifier_ensemble = WeightedChartClassifier(model_manager)
                for idx in needs_classification:
                    asset = assets[idx]
                    try:
                        img_buf = asset.image_buffer
                        if img_buf is None and asset.image_path.exists():
                            img_buf = cv2.imread(str(asset.image_path))

                        if img_buf is not None and img_buf.size > 0:
                            types, top_conf = classifier_ensemble.classify_image_with_conf(
                                img_buf, top_k=2, image_path=asset.image_path
                            )
                            conf_val = max(0.0, min(1.0, float(top_conf or 0.0)))
                            is_low = conf_val < LOW_CONFIDENCE_THRESHOLD
                            conf_dict = {
                                'classification': conf_val,
                                'detection': 0.0,
                                'average': conf_val,
                                'is_low_confidence': is_low,
                                'source': 'preliminary_classification_only',
                                'chart_types': types,
                            }
                            assets[idx] = asset._replace(confidence_info=conf_dict)
                    except Exception as c_exc:
                        logger.debug("Upfront classification failed for asset %s: %s", asset.figure_id, c_exc)

                # Persist updated confidence_info back to metadata JSON
                metadata_file = render_dir / f"{pdf_path.stem}_processing_metadata.json"
                if metadata_file.exists():
                    try:
                        import json
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            meta_data = json.load(f)
                        charts_list = meta_data.get("charts", [])
                        for idx, a in enumerate(assets):
                            if idx < len(charts_list) and a.confidence_info:
                                charts_list[idx]["confidence_info"] = a.confidence_info
                                if "chart_types" in a.confidence_info and a.confidence_info["chart_types"]:
                                    charts_list[idx]["preliminary_type"] = a.confidence_info["chart_types"][0]
                        with open(metadata_file, "w", encoding="utf-8") as f:
                            json.dump(meta_data, f, ensure_ascii=False, indent=2)
                        logger.debug("Persisted upfront classification into %s", metadata_file.name)
                    except Exception as p_exc:
                        logger.debug("Failed saving updated metadata to %s: %s", metadata_file.name, p_exc)
            except Exception as exc:
                logger.debug("Failed to run upfront classification ensemble: %s", exc)
        else:
            # Seed the classification cache from already-loaded metadata so batch pipeline gets cache hits
            try:
                from core.ensemble_classifier import WeightedChartClassifier
                for a in assets:
                    if a.confidence_info and 'chart_types' in a.confidence_info and 'classification' in a.confidence_info:
                        WeightedChartClassifier.set_cached_result(
                            a.image_path,
                            a.confidence_info['chart_types'],
                            a.confidence_info['classification'],
                        )
            except Exception as seed_exc:
                logger.debug("Failed seeding classification cache from metadata: %s", seed_exc)

    logger.info("PDF %s expanded to %d chart(s)", pdf_path.name, len(assets))
    return assets


def _resolve_single_file(
    file_path: Path,
    render_dir: Path,
    input_type: str,
    high_res_dpi: int = 200,
    min_chart_width: int = 300,
    min_chart_height: int = 200,
    progress_callback: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[Any] = None,
    failure_callback: Optional[Callable[[Path, str], None]] = None,
    model_manager: Optional[Any] = None,
    force_reextract: bool = False,
    check_cache_only: bool = False,
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
            cancel_event=cancel_event, failure_callback=failure_callback,
            model_manager=model_manager,
            force_reextract=force_reextract,
            check_cache_only=check_cache_only,
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
    high_res_dpi: int = 200,
    min_chart_width: int = 300,
    min_chart_height: int = 200,
    progress_callback: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[Any] = None,
    failure_callback: Optional[Callable[[Path, str], None]] = None,
    model_manager: Optional[Any] = None,
    force_reextract: bool = False,
    check_cache_only: bool = False,
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
                cancel_event=cancel_event, failure_callback=failure_callback,
                model_manager=model_manager,
                force_reextract=force_reextract,
                check_cache_only=check_cache_only,
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
    high_res_dpi: int = 200,
    min_chart_width: int = 300,
    min_chart_height: int = 200,
    progress_callback: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[Any] = None,
    failure_callback: Optional[Callable[[Path, str], None]] = None,
    model_manager: Optional[Any] = None,
    force_reextract: bool = False,
    check_cache_only: bool = False,
) -> List[ResolvedAsset]:
    """Resolve an input path into a flat, ordered list of ResolvedAsset objects."""
    input_path = Path(input_path)
    render_dir = Path(render_dir)

    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        return []

    if input_path.is_file():
        return _resolve_single_file(
            input_path, render_dir, input_type,
            high_res_dpi, min_chart_width, min_chart_height,
            progress_callback, cancel_event=cancel_event,
            failure_callback=failure_callback,
            model_manager=model_manager,
            force_reextract=force_reextract,
            check_cache_only=check_cache_only,
        )

    if input_path.is_dir():
        return _resolve_directory(
            input_path, render_dir, input_type,
            high_res_dpi, min_chart_width, min_chart_height,
            progress_callback, cancel_event=cancel_event,
            failure_callback=failure_callback,
            model_manager=model_manager,
            force_reextract=force_reextract,
            check_cache_only=check_cache_only,
        )

    logger.error("Input path is neither a file nor a directory: %s", input_path)



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
