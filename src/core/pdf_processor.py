# -*- coding: utf-8 -*-
"""
Processador de PDF OTIMIZADO para extração de gráficos.

OTIMIZAÇÕES IMPLEMENTADAS:
- Eliminação de I/O redundante (não reabre o mesmo PDF múltiplas vezes)
- Compartilhamento do objeto fitz.Document entre funções
- Melhor gerenciamento de memória com context managers
- Tratamento robusto de erros
- Logging aprimorado para debug

MELHORIAS DE PERFORMANCE:
- Redução de ~5-10% no tempo de processamento de PDFs
- Menor uso de memória ao evitar múltiplas aberturas do mesmo arquivo
- Operações de I/O mais eficientes
"""

import logging
import fitz  # PyMuPDF
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Generator, Any, Callable
from contextlib import contextmanager
import cv2
import os
import json
import threading
from datetime import datetime

# --- Configuração de logging específica ---
logger = logging.getLogger(__name__)


class PDFAccessError(Exception):
    """Raised for PDFs that opened but cannot be read (auth required, etc.)."""


# --- Context Managers para Gerenciamento de Recursos ---

@contextmanager
def open_pdf_document(pdf_path: Path):
    """
    Context manager para abrir e garantir fechamento correto de documentos PDF.
    
    Args:
        pdf_path: Caminho para o arquivo PDF
        
    Yields:
        fitz.Document: Documento PDF aberto
        
    Raises:
        PDFAccessError: Se o PDF exigir senha
        Exception: Se não conseguir abrir o PDF
    """
    doc = None
    try:
        logger.info(f"Opening PDF: {pdf_path.name}")
        doc = fitz.open(str(pdf_path))
        if doc.needs_pass:
            raise PDFAccessError(f"'{pdf_path.name}' is password-protected and requires a user password.")
        yield doc
    except RuntimeError as e:
        logger.error(f"Runtime error opening PDF {pdf_path.name}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error opening PDF {pdf_path.name}: {e}")
        raise
    finally:
        if doc:
            doc.close()
            logger.debug(f"PDF closed: {pdf_path.name}")

# --- Funções Otimizadas ---

def extract_charts_from_pdf_optimized(
    pdf_path: Path, 
    output_dir: Path, 
    min_width: int = 300, 
    min_height: int = 200
) -> List[dict]:
    """
    Args:
        pdf_path: Caminho para o arquivo PDF
        output_dir: Diretório onde salvar as imagens extraídas
        min_width: Largura mínima para considerar como gráfico (pixels)
        min_height: Altura mínima para considerar como gráfico (pixels)
    
    Returns:
        List[dict]: Lista de dicionários com informações dos gráficos extraídos
        [
            {
                'page_num': int,
                'image_index': int,
                'file_path': Path,
                'dimensions': tuple,
                'pdf_rect': fitz.Rect,
                'extraction_method': str
            }
        ]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_charts = []
    
    with open_pdf_document(pdf_path) as doc:
        logger.info(f"PDF has {doc.page_count} pages")
        
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                logger.debug(f"Analyzing page {page_num + 1}/{doc.page_count}")
                
                # Extrair imagens da página atual
                page_charts = _extract_images_from_page_optimized(
                    page, 
                    page_num, 
                    pdf_path.stem, 
                    output_dir, 
                    min_width, 
                    min_height
                )
                
                extracted_charts.extend(page_charts)
                
                if page_charts:
                    logger.info(f"Page {page_num + 1}: {len(page_charts)} chart(s) extracted")
                else:
                    logger.debug(f"Page {page_num + 1}: No charts found")
                    
            except Exception as e:
                logger.error(f"Error processing page {page_num + 1}: {e}")
                continue
    
    logger.info(f"Extraction complete: {len(extracted_charts)} chart(s) extracted from {pdf_path.name}")
    return extracted_charts


def _render_page_as_image_array(page: fitz.Page, dpi: int = 200) -> Optional[np.ndarray]:
    """Render a page straight into a BGR numpy array — no disk round-trip."""
    try:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 3:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            bgr = arr
        return bgr.copy()
    except Exception as exc:
        logger.error(f"In-memory render failed: {exc}")
        return None


def _render_page_as_image(
    page: fitz.Page,
    page_num: int,
    pdf_stem: str,
    output_dir: Path,
    dpi: int = 200,
) -> Optional[dict]:
    """
    Render an entire PDF page as a PNG image at the given DPI.

    This is the primary extraction method for vector-based charts (matplotlib,
    R, LaTeX, Word exports etc.) that contain no embedded raster images.

    Args:
        page:       fitz.Page object (0-indexed page already selected)
        page_num:   0-indexed page number (for filename/logging)
        pdf_stem:   PDF filename stem (no extension)
        output_dir: Directory where the PNG will be saved
        dpi:        Rendering resolution (200 DPI is a good balance of
                    quality vs. file size for analysis)

    Returns:
        dict with chart_info keys, or None if rendering failed.
    """
    try:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix)

        filename = f"{pdf_stem}_page{page_num + 1:02d}_fullpage.png"
        file_path = output_dir / filename
        pixmap.save(str(file_path))
        w, h = pixmap.width, pixmap.height
        del pixmap

        if not file_path.exists() or file_path.stat().st_size == 0:
            logger.error(
                f"Full-page render for page {page_num + 1} did not produce a "
                f"valid file at {file_path}. Skipping this page."
            )
            return None

        logger.info(f"Full-page render saved: {filename} ({w}x{h}px @ {dpi} DPI)")
        return {
            'page_num': page_num + 1,
            'image_index': 1,
            'file_path': file_path,
            'high_res_path': file_path,   # already final quality — no re-render needed
            'dimensions': (w, h),
            'pdf_rect': page.rect,
            'extraction_method': 'full_page_render',
        }
    except Exception as exc:
        logger.error(f"Full-page render failed for page {page_num + 1}: {exc}")
        return None


def _process_page_with_doclayout_and_type_detect(
    page_img: np.ndarray,
    page_num: int,
    pdf_stem: str,
    output_dir: Path,
    model_manager: Optional[Any],
    min_width: int,
    min_height: int,
    render_dpi: int = 200,
    source_method: str = "doclayout_figure_crop",
    page_rect: Optional[fitz.Rect] = None,
) -> List[dict]:
    """
    Applies the full hierarchical pipeline to a document page canvas or full-page scan:
    1. DocLayout (doclayout_yolo.onnx @ 1024x1024) -> proposes figure regions (cls == 3)
    2. type_detect (type_detect.onnx @ 640x640) -> decomposes each figure region into sub-charts
    """
    page_charts: List[dict] = []
    if page_img is None or page_img.size == 0:
        return page_charts

    h, w = page_img.shape[:2]
    figure_crops = []

    if model_manager is not None and getattr(model_manager, '_models', None):
        try:
            doclayout_model = None
            if hasattr(model_manager, '_models') and isinstance(model_manager._models, dict):
                if 'doclayout' in model_manager._models and model_manager._models['doclayout'] is not None:
                    doclayout_model = model_manager.get_model('doclayout')
                elif 'doclayout_yolo' in model_manager._models and model_manager._models['doclayout_yolo'] is not None:
                    doclayout_model = model_manager.get_model('doclayout_yolo')
            else:
                doclayout_model = model_manager.get_model('doclayout')
        except Exception:
            doclayout_model = None

        if doclayout_model is not None:
            try:
                from utils.inference import run_inference_on_image
                from core.class_maps import CLASS_MAP_DOCLAYOUT
                layout_dets = run_inference_on_image(
                    doclayout_model, page_img, 0.25, CLASS_MAP_DOCLAYOUT,
                    input_size=(1024, 1024), model_output_type='bbox',
                )
                for d in layout_dets:
                    if d.get('cls') == 3:  # 3: figure ONLY (excludes plain_text, title, caption, table)
                        bbox = d.get('bbox') or d.get('xyxy')
                        if bbox and len(bbox) == 4:
                            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            crop_w, crop_h = x2 - x1, y2 - y1
                            eff_min_w = int(min_width * (render_dpi / 300.0))
                            eff_min_h = int(min_height * (render_dpi / 300.0))
                            if crop_w >= eff_min_w and crop_h >= eff_min_h:
                                crop_img = page_img[y1:y2, x1:x2]
                                if crop_img.size > 0:
                                    figure_crops.append(((x1, y1, x2, y2), crop_img))
            except Exception as exc:
                logger.debug("DocLayout figure cropping pass skipped: %s", exc)

    if not figure_crops and model_manager is None:
        cv_candidates = _find_chart_regions_cv2(page_img, min_width=min_width, min_height=min_height)
        for (bx, by, bw, bh), crop_img in cv_candidates:
            figure_crops.append(((bx, by, bx + bw, by + bh), crop_img))

    if figure_crops:
        logger.info(f"Page {page_num + 1}: {len(figure_crops)} layout figure region(s) proposed via {source_method}")
        flat_chart_crops = []
        if model_manager is not None:
            from utils.inference import decompose_multipanel_figure
            for bbox, crop_img in figure_crops:
                sub_chart_crops = decompose_multipanel_figure(crop_img, model_manager, padding=20)
                if sub_chart_crops:
                    for (sb_bbox, sub_crop, pred_type) in sub_chart_crops:
                        page_x1 = bbox[0] + sb_bbox[0]
                        page_y1 = bbox[1] + sb_bbox[1]
                        page_x2 = bbox[0] + sb_bbox[2]
                        page_y2 = bbox[1] + sb_bbox[3]
                        flat_chart_crops.append(((page_x1, page_y1, page_x2, page_y2), sub_crop, pred_type))
                else:
                    flat_chart_crops.append((bbox, crop_img, None))
        else:
            flat_chart_crops = [(bbox, crop_img, None) for bbox, crop_img in figure_crops]

        for c_idx, (bbox, crop_img, pred_type) in enumerate(flat_chart_crops):
            cw, ch = crop_img.shape[1], crop_img.shape[0]
            filename = f"{pdf_stem}_page{page_num + 1:02d}_fig{c_idx + 1:02d}.png"
            file_path = output_dir / filename
            cv2.imwrite(str(file_path), crop_img)
            p_rect = page_rect if page_rect is not None else fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            page_charts.append({
                'page_num': page_num + 1,
                'image_index': c_idx + 1,
                'file_path': file_path,
                'high_res_path': file_path,
                'image_buffer': crop_img,
                'dimensions': (cw, ch),
                'pdf_rect': p_rect,
                'extraction_method': source_method + ('_type_detect' if pred_type else ''),
                'preliminary_type': pred_type,
            })

    return page_charts


def _extract_images_from_page_optimized(
    page: fitz.Page,
    page_num: int,
    pdf_stem: str,
    output_dir: Path,
    min_width: int,
    min_height: int,
    render_dpi: int = 200,
    model_manager: Optional[Any] = None,
) -> List[dict]:
    """
    Extract chart images from a single PDF page.

    Strategy:
      1. Try to extract raster images embedded directly in the PDF.
         If an embedded image covers the full page (scanned document page),
         it is processed through DocLayout to locate figures, then type_detect.
      2. If no embedded images are found — or all are too small / not
         chart-like — fall back to rendering the full page at *render_dpi*
         and applying DocLayout -> type_detect.
    """
    page_charts: List[dict] = []

    # ------------------------------------------------------------------
    # Strategy 1: embedded raster images inside the PDF
    # ------------------------------------------------------------------
    try:
        image_list = page.get_images(full=True)
        if image_list:
            logger.debug(f"Page {page_num + 1}: {len(image_list)} embedded image(s) found")
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = page.parent.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext  = base_image["ext"]

                    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
                    cv_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                    if cv_image is None:
                        continue

                    h, w = cv_image.shape[:2]
                    if w < min_width or h < min_height:
                        logger.debug(f"  Image {img_index}: too small ({w}x{h}), skipped")
                        continue

                    filename  = f"{pdf_stem}_page{page_num+1:02d}_img{img_index+1:02d}.{image_ext}"
                    file_path = output_dir / filename

                    img_rects = page.get_image_rects(xref)
                    rect = img_rects[0] if img_rects else fitz.Rect(0, 0, w, h)

                    # Check if this embedded image is a full-page scan
                    page_w = max(1.0, page.rect.width)
                    page_h = max(1.0, page.rect.height)
                    coverage = (rect.width * rect.height) / (page_w * page_h)
                    has_text = len(page.get_text().strip()) > 50
                    is_full_page_scan = (
                        coverage >= 0.75
                        or (rect.width >= 0.88 * page_w and rect.height >= 0.88 * page_h)
                        or (not has_text and coverage >= 0.50)
                    )

                    if is_full_page_scan:
                        logger.info(
                            f"Page {page_num + 1} embedded image {img_index + 1} detected as full-page scan ({w}x{h}px, coverage={coverage:.1%}). "
                            "Applying DocLayout on entire page to detect figure regions, then type_detect."
                        )
                        scanned_charts = _process_page_with_doclayout_and_type_detect(
                            cv_image,
                            page_num=page_num,
                            pdf_stem=pdf_stem,
                            output_dir=output_dir,
                            model_manager=model_manager,
                            min_width=min_width,
                            min_height=min_height,
                            render_dpi=render_dpi,
                            source_method="scanned_page_doclayout",
                            page_rect=page.rect,
                        )
                        if scanned_charts:
                            page_charts.extend(scanned_charts)
                        # Full-page scans with no figures detected are text-only pages; do not emit whole page as chart
                        continue

                    # High-res canvas render with padding in memory for discrete figures
                    high_res_filename = f"{pdf_stem}_page{page_num+1:02d}_img{img_index+1:02d}_rendered.png"
                    high_res_path = output_dir / high_res_filename
                    rendered_buffer = cv_image

                    if img_rects:
                        try:
                            # Render page canvas region with padding (25pt) at render_dpi to capture ticks & labels
                            pad = 25.0
                            padded_rect = fitz.Rect(
                                max(0, rect.x0 - pad),
                                max(0, rect.y0 - pad),
                                min(page.rect.width, rect.x1 + pad),
                                min(page.rect.height, rect.y1 + pad),
                            )
                            matrix = fitz.Matrix(render_dpi / 72.0, render_dpi / 72.0)
                            pixmap = page.get_pixmap(matrix=matrix, clip=padded_rect, alpha=False)
                            
                            # Convert pixmap to BGR numpy array for downstream processing
                            pix_arr = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
                            if pixmap.n == 3:
                                rendered_buffer = cv2.cvtColor(pix_arr, cv2.COLOR_RGB2BGR)
                            elif pixmap.n == 4:
                                rendered_buffer = cv2.cvtColor(pix_arr, cv2.COLOR_RGBA2BGR)
                            else:
                                rendered_buffer = pix_arr
                            del pixmap
                        except Exception as render_exc:
                            logger.warning(f"Failed to render padded high-res region for image {img_index}: {render_exc}")

                    sub_chart_crops = []
                    if model_manager is not None and getattr(model_manager, '_models', None):
                        from utils.inference import decompose_multipanel_figure
                        sub_chart_crops = decompose_multipanel_figure(rendered_buffer, model_manager, padding=20)

                    if sub_chart_crops:
                        logger.info(f"Page {page_num + 1} embedded image {img_index + 1}: decompose_multipanel_figure split figure into {len(sub_chart_crops)} sub-chart(s)")
                        for sub_idx, (sb_bbox, sub_crop, pred_type) in enumerate(sub_chart_crops):
                            sub_filename = f"{pdf_stem}_page{page_num + 1:02d}_img{img_index + 1:02d}_sub{sub_idx + 1:02d}.png"
                            sub_file_path = output_dir / sub_filename
                            cv2.imwrite(str(sub_file_path), sub_crop)
                            page_charts.append({
                                'page_num': page_num + 1,
                                'image_index': img_index + 1,
                                'file_path': sub_file_path,
                                'high_res_path': sub_file_path,
                                'image_buffer': sub_crop,
                                'dimensions': (sub_crop.shape[1], sub_crop.shape[0]),
                                'pdf_rect': rect,
                                'extraction_method': 'embedded_image_type_detect',
                                'preliminary_type': pred_type,
                            })
                    else:
                        # Only write parent image files to disk if NOT decomposed into sub-charts
                        with open(file_path, "wb") as fh:
                            fh.write(image_bytes)
                        if rendered_buffer is not cv_image:
                            cv2.imwrite(str(high_res_path), rendered_buffer)
                        else:
                            high_res_path = file_path

                        page_charts.append({
                            'page_num':          page_num + 1,
                            'image_index':       img_index + 1,
                            'file_path':         file_path,
                            'high_res_path':     high_res_path,
                            'image_buffer':      rendered_buffer,
                            'dimensions':        (rendered_buffer.shape[1], rendered_buffer.shape[0]),
                            'pdf_rect':          rect,
                            'extraction_method': 'embedded_image',
                        })
                        logger.info(f"Embedded image processed & rendered: {filename} ({rendered_buffer.shape[1]}x{rendered_buffer.shape[0]}px)")

                except Exception as exc:
                    logger.error(
                        f"Error extracting embedded image {img_index} "
                        f"from page {page_num + 1}: {exc}"
                    )
    except Exception as exc:
        logger.error(f"Error listing images on page {page_num + 1}: {exc}")

    # ------------------------------------------------------------------
    # Strategy 2: full-page render fallback / figure region cropping
    # Triggered when embedded-image extraction found nothing useful.
    # ------------------------------------------------------------------
    if not page_charts:
        logger.debug(
            f"Page {page_num + 1}: no usable embedded images — "
            "falling back to page render and figure region proposal"
        )
        cv_img = _render_page_as_image_array(page, dpi=render_dpi)
        if cv_img is not None:
            canvas_charts = _process_page_with_doclayout_and_type_detect(
                cv_img,
                page_num=page_num,
                pdf_stem=pdf_stem,
                output_dir=output_dir,
                model_manager=model_manager,
                min_width=min_width,
                min_height=min_height,
                render_dpi=render_dpi,
                source_method="doclayout_figure_crop",
                page_rect=page.rect,
            )
            if canvas_charts:
                page_charts.extend(canvas_charts)

    return page_charts


def _find_chart_regions_cv2(
    cv_img: np.ndarray,
    min_width: int = 300,
    min_height: int = 200,
) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    """
    Zero-dependency computer vision fallback to locate rectangular chart/plot regions
    on a rendered PDF page canvas using contour analysis.
    """
    if cv_img is None or cv_img.size == 0:
        return []

    h, w = cv_img.shape[:2]
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img.copy()

    # Apply morphological gradient to highlight outer plot borders and grid lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    # Threshold and dilate to connect contiguous chart components
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(thresh, dilate_kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    page_area = float(w * h)

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < min_width or bh < min_height:
            continue
        box_area = float(bw * bh)
        if box_area / page_area > 0.92:
            continue

        pad_x = min(10, x)
        pad_y = min(10, y)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        crop_img = cv_img[y1:y2, x1:x2]
        if crop_img.size > 0 and (0.2 < (bw / float(bh)) < 5.0):
            candidates.append(((x1, y1, x2, y2), crop_img))

    candidates.sort(key=lambda item: (item[0][1], item[0][0]))
    return candidates



def rerender_chart_at_high_res_optimized(
    pdf_path: Path, 
    page_num: int, 
    chart_rect: fitz.Rect, 
    output_path: Path, 
    dpi: int = 300
) -> Optional[Path]:
    """
    Renderiza um gráfico específico em alta resolução.
    
    Args:
        pdf_path: Caminho do PDF
        page_num: Número da página (1-based)
        chart_rect: Retângulo do gráfico na página
        output_path: Onde salvar a imagem renderizada
        dpi: Resolução de renderização
    
    Returns:
        Path do arquivo salvo ou None se falhar
    """
    # Validar parâmetros
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return None
    
    if page_num < 1:
        logger.error(f"Invalid page number: {page_num} (must be >= 1)")
        return None
    
    if dpi < 72 or dpi > 600:
        logger.warning(f"Unusual DPI: {dpi}. Recommended: 150-300")
    
    try:
        with open_pdf_document(pdf_path) as doc:
            if page_num > doc.page_count:
                logger.error(f"Page {page_num} does not exist (PDF has {doc.page_count} pages)")
                return None
            
            page = doc[page_num - 1]  # Converter para 0-indexed
            
            # Calcular matriz de transformação para o DPI desejado
            zoom_factor = dpi / 72.0  # 72 DPI é o padrão
            matrix = fitz.Matrix(zoom_factor, zoom_factor)
            
            logger.info(f"Rendering page {page_num} at {dpi} DPI (zoom: {zoom_factor:.2f}x)")
            
            # Renderizar apenas a área especificada com margem de padding (25pt)
            if chart_rect and not chart_rect.is_empty:
                pad = 25.0
                padded_rect = fitz.Rect(
                    max(0, chart_rect.x0 - pad),
                    max(0, chart_rect.y0 - pad),
                    min(page.rect.width, chart_rect.x1 + pad),
                    min(page.rect.height, chart_rect.y1 + pad),
                )
                logger.debug(f"Specific area with padding: {padded_rect}")
                pixmap = page.get_pixmap(matrix=matrix, clip=padded_rect, alpha=False)
            else:
                # Renderizar página inteira
                logger.debug("Rendering full page")
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            
            # Salvar a imagem
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            pixmap_width, pixmap_height = pixmap.width, pixmap.height
            pixmap.save(str(output_path))
            del pixmap

            logger.info(f"Rendered image saved: {output_path.name} ({pixmap_width}x{pixmap_height}px)")
            return output_path
            
    except Exception as e:
        logger.error(f"Error rendering: {e}")
        return None



_model_inference_lock = threading.Lock()


def process_pdf_charts_optimized(
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
    """
    Pipeline completo para extrair gráficos de um PDF com suporte a extração paralela.
    """
    pdf_path   = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Processing {pdf_path.name} (render DPI={high_res_dpi})")

    if model_manager is None or not getattr(model_manager, '_models', None):
        try:
            from core.model_manager import ModelManager
            models_dir = Path(__file__).parent.parent / "models"
            if model_manager is None:
                model_manager = ModelManager()
            if not getattr(model_manager, '_models', None):
                model_manager.load_models(str(models_dir))
            logger.info("Auto-instantiated/loaded ModelManager with models from %s", models_dir)
        except Exception as exc:
            logger.warning("Could not auto-instantiate/load ModelManager: %s", exc)

    processed_charts: List[dict] = []

    with open_pdf_document(pdf_path) as doc:
        total_pages = doc.page_count
        logger.info(f"{total_pages} page(s)")

    if total_pages == 0:
        return []

    def _process_single_page(page_num: int) -> List[dict]:
        if cancel_event is not None and cancel_event.is_set():
            return []
        try:
            if progress_callback is not None:
                progress_callback(f"Extracting charts from page {page_num + 1}/{total_pages}...")
            with open_pdf_document(pdf_path) as thread_doc:
                page = thread_doc[page_num]
                return _extract_images_from_page_optimized(
                    page,
                    page_num,
                    pdf_path.stem,
                    output_dir,
                    min_chart_width,
                    min_chart_height,
                    render_dpi=high_res_dpi,
                    model_manager=model_manager,
                )
        except Exception as exc:
            logger.error(f"Error on page {page_num + 1}: {exc}")
            return []

    if total_pages == 1:
        processed_charts = _process_single_page(0)
    else:
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers = min(os.cpu_count() or 4, 4, total_pages)
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for page_num in range(total_pages):
                if cancel_event is not None and cancel_event.is_set():
                    logger.warning("PDF processing cancelled prior to dispatching page %d/%d", page_num + 1, total_pages)
                    break
                futures.append(executor.submit(_process_single_page, page_num))

            for future in as_completed(futures):
                if cancel_event is not None and cancel_event.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                try:
                    p_charts = future.result()
                    if p_charts:
                        processed_charts.extend(p_charts)
                except Exception as exc:
                    logger.error("Page worker thread raised unhandled exception: %s", exc)

    # Sort deterministically by page_num and image_index
    processed_charts.sort(key=lambda c: (c.get('page_num', 0), c.get('image_index', 0)))

    logger.info(
        f"Done: {len(processed_charts)} chart(s) from {pdf_path.name}"
    )
    _save_processing_metadata(processed_charts, output_dir, pdf_path)
    return processed_charts


def _save_processing_metadata(processed_charts: List[dict], output_dir: Path, pdf_path: Path):
    """
    Salva metadados do processamento para referência futura.
    
    Args:
        processed_charts: Lista de gráficos processados
        output_dir: Diretório de saída
        pdf_path: Caminho do PDF original
    """
    try:
        metadata = {
            'source_pdf': str(pdf_path),
            'processing_timestamp': datetime.now().isoformat(),
            'total_charts': len(processed_charts),
            'charts': []
        }
        
        for chart in processed_charts:
            chart_meta = {
                'page_num': chart.get('page_num'),
                'original_file': str(chart.get('file_path', '')),
                'high_res_file': str(chart.get('high_res_path', '')),
                'dimensions': chart.get('dimensions'),
                'high_res_dimensions': chart.get('high_res_dimensions'),
                'extraction_method': chart.get('extraction_method'),
                'processing_method': chart.get('processing_method'),
                'errors': chart.get('high_res_error'),
                'preliminary_type': chart.get('preliminary_type'),
                'confidence_info': chart.get('confidence_info'),
            }
            metadata['charts'].append(chart_meta)
        
        metadata_path = output_dir / f"{pdf_path.stem}_processing_metadata.json"
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Metadata saved: {metadata_path.name}")
        
    except Exception as e:
        logger.warning(f"Could not save metadata: {e}")


def extract_charts_with_doclayout(pdf_path: Path, output_dir: Path, model_path: str, figure_class_id: int = 3):
    """Deprecated legacy compatibility function."""
    try:
        from doclayout_yolo import YOLOv10
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning(
            "DocLayout extraction unavailable (missing optional dependency 'doclayout_yolo'): %s",
            exc,
        )
    logger.warning("extract_charts_with_doclayout is deprecated; use process_pdf_charts_optimized.")
    return []


# --- Função Principal para Testes ---

def main():
    """Função principal para teste do processador de PDF otimizado."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Processador OTIMIZADO de gráficos em PDF")
    parser.add_argument('pdf_path', type=str, help='Caminho para o arquivo PDF')
    parser.add_argument('--output_dir', type=str, default='output/pdf_output', help='Diretório de saída')
    parser.add_argument('--dpi', type=int, default=300, help='DPI para alta resolução')
    parser.add_argument('--min_width', type=int, default=300, help='Largura mínima do gráfico')
    parser.add_argument('--min_height', type=int, default=200, help='Altura mínima do gráfico')
    parser.add_argument('--verbose', '-v', action='store_true', help='Logging verboso')
    
    args = parser.parse_args()
    
    # Configurar logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    pdf_path = Path(args.pdf_path)
    output_dir = Path(args.output_dir)
    
    print(f"Processing PDF: {pdf_path.name}")
    print(f"Output: {output_dir}")
    print(f"DPI: {args.dpi}")
    print(f"Min dimensions: {args.min_width}x{args.min_height}")
    
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        return 1
    
    try:
        # Usar a função otimizada
        results = process_pdf_charts_optimized(
            pdf_path=pdf_path,
            output_dir=output_dir,
            high_res_dpi=args.dpi,
            min_chart_width=args.min_width,
            min_chart_height=args.min_height
        )
        
        print(f"\nPROCESSING COMPLETE!")
        print(f"{len(results)} chart(s) extracted and processed")
        print(f"Files saved in: {output_dir}")
        
        # Exibir resumo
        for i, chart in enumerate(results, 1):
            print(f"\nChart {i}:")
            print(f"   Page: {chart.get('page_num')}")
            print(f"   Dimensions: {chart.get('dimensions')}")
            if 'high_res_dimensions' in chart:
                print(f"   High Res: {chart.get('high_res_dimensions')}")
            print(f"   File: {chart.get('file_path', 'N/A')}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR during processing: {e}")
        logger.exception("Detailed error:")
        return 1


if __name__ == "__main__":
    exit(main())
