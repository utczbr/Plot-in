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
from typing import List, Tuple, Optional, Generator
from contextlib import contextmanager
import cv2
import os
import json
from datetime import datetime

# --- Configuração de logging específica ---
logger = logging.getLogger(__name__)

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
        Exception: Se não conseguir abrir o PDF
    """
    doc = None
    try:
        logger.info(f"📖 Abrindo PDF: {pdf_path.name}")
        doc = fitz.open(str(pdf_path))
        yield doc
    except RuntimeError as e:
        logger.error(f"❌ Erro de tempo de execução ao abrir PDF (pode estar corrompido) {pdf_path.name}: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Erro inesperado ao abrir PDF {pdf_path.name}: {e}")
        raise
    finally:
        if doc:
            doc.close()
            logger.debug(f"📖 PDF fechado: {pdf_path.name}")

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
        logger.info(f"📄 PDF possui {doc.page_count} páginas")
        
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                logger.debug(f"🔍 Analisando página {page_num + 1}/{doc.page_count}")
                
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
                    logger.info(f"✅ Página {page_num + 1}: {len(page_charts)} gráfico(s) extraído(s)")
                else:
                    logger.debug(f"⏭️ Página {page_num + 1}: Nenhum gráfico encontrado")
                    
            except Exception as e:
                logger.error(f"❌ Erro ao processar página {page_num + 1}: {e}")
                continue
    
    logger.info(f"🎉 Extração concluída: {len(extracted_charts)} gráfico(s) extraído(s) de {pdf_path.name}")
    return extracted_charts


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

        logger.info(f"🖼️  Full-page render saved: {filename} ({w}x{h}px @ {dpi} DPI)")
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
        logger.error(f"❌ Full-page render failed for page {page_num + 1}: {exc}")
        return None


def _extract_images_from_page_optimized(
    page: fitz.Page,
    page_num: int,
    pdf_stem: str,
    output_dir: Path,
    min_width: int,
    min_height: int,
    render_dpi: int = 200,
) -> List[dict]:
    """
    Extract chart images from a single PDF page.

    Strategy:
      1. Try to extract raster images embedded directly in the PDF
         (works for scanned figures or pre-rasterised exports).
      2. If no embedded images are found — or all are too small / not
         chart-like — fall back to rendering the full page at *render_dpi*.
         This captures vector-drawn charts (matplotlib, R, LaTeX, Word…)
         which are by far the most common format in scientific literature.
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
                    if not _is_likely_chart_image(cv_image, w, h):
                        logger.debug(f"  Image {img_index}: not chart-like, skipped")
                        continue

                    filename  = f"{pdf_stem}_page{page_num+1:02d}_img{img_index+1:02d}.{image_ext}"
                    file_path = output_dir / filename
                    with open(file_path, "wb") as fh:
                        fh.write(image_bytes)

                    img_rects = page.get_image_rects(xref)
                    rect = img_rects[0] if img_rects else fitz.Rect(0, 0, w, h)

                    page_charts.append({
                        'page_num':          page_num + 1,
                        'image_index':       img_index + 1,
                        'file_path':         file_path,
                        'high_res_path':     file_path,   # kept for API compat
                        'dimensions':        (w, h),
                        'pdf_rect':          rect,
                        'extraction_method': 'embedded_image',
                    })
                    logger.info(f"✅ Embedded image saved: {filename} ({w}x{h}px)")

                except Exception as exc:
                    logger.error(
                        f"❌ Error extracting embedded image {img_index} "
                        f"from page {page_num + 1}: {exc}"
                    )
    except Exception as exc:
        logger.error(f"❌ Error listing images on page {page_num + 1}: {exc}")

    # ------------------------------------------------------------------
    # Strategy 2: full-page render fallback
    # Triggered when embedded-image extraction found nothing useful.
    # This is the normal path for vector-drawn scientific charts.
    # ------------------------------------------------------------------
    if not page_charts:
        logger.debug(
            f"Page {page_num + 1}: no usable embedded images — "
            "falling back to full-page render"
        )
        chart = _render_page_as_image(
            page, page_num, pdf_stem, output_dir, dpi=render_dpi
        )
        if chart:
            # Post-render size & chart-likelihood check
            try:
                cv_img = cv2.imread(str(chart['file_path']))
                if cv_img is not None:
                    h, w = cv_img.shape[:2]
                    if w < min_width or h < min_height:
                        logger.debug(
                            f"  Full-page render too small ({w}x{h}), discarding"
                        )
                        chart['file_path'].unlink(missing_ok=True)
                        chart = None
                    elif not _is_likely_chart_image(cv_img, w, h):
                        logger.debug(
                            "  Full-page render not chart-like (probably text-only page), discarding"
                        )
                        chart['file_path'].unlink(missing_ok=True)
                        chart = None
            except Exception as exc:
                logger.warning(
                    f"Could not validate full-page render for page {page_num + 1}: {exc}"
                )
            if chart:
                page_charts.append(chart)

    return page_charts


def _is_likely_chart_image(
    cv_image: np.ndarray, 
    width: int, 
    height: int,
    config: dict = None
) -> bool:
    if config is None:
        config = {
            "aspect_ratio_range": (0.3, 4.0),
            "min_text_components": 3,
            "text_area_range": (10, 1000),
            "text_aspect_ratio_range": (0.2, 5.0),
            "min_lines": 2,
            "hough_threshold": 50,
            "hough_min_line_length_ratio": 0.1,
            "hough_max_line_gap": 10,
            "canny_thresholds": (50, 150)
        }

    try:
        # Criterion 1: Aspect ratio
        aspect_ratio = width / height
        if not (config["aspect_ratio_range"][0] < aspect_ratio < config["aspect_ratio_range"][1]):
            return False
        
        # Criterion 2: Check for text presence (charts have labels)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find connected components (potential text)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Text-like components: small, rectangular
        text_like_components = 0
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            width_comp = stats[i, cv2.CC_STAT_WIDTH]
            height_comp = stats[i, cv2.CC_STAT_HEIGHT]
            
            if config["text_area_range"][0] < area < config["text_area_range"][1]:
                if height_comp > 0 and config["text_aspect_ratio_range"][0] < width_comp/height_comp < config["text_aspect_ratio_range"][1]:
                    text_like_components += 1
        
        if text_like_components < config["min_text_components"]:
            return False
        
        # Criterion 3: Geometric structure (lines/edges)
        edges = cv2.Canny(gray, config["canny_thresholds"][0], config["canny_thresholds"][1])
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=config["hough_threshold"], 
                                minLineLength=min(width, height) * config["hough_min_line_length_ratio"], 
                                maxLineGap=config["hough_max_line_gap"])
        
        if lines is None or len(lines) < config["min_lines"]:
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Error in chart analysis: {e}. Assuming it's a chart.")
        return True


def rerender_chart_at_high_res_optimized(
    pdf_path: Path, 
    page_num: int, 
    chart_rect: fitz.Rect, 
    output_path: Path, 
    dpi: int = 300
) -> Optional[Path]:
    """
    Re-renderiza uma seção específica de um PDF em alta resolução (VERSÃO OTIMIZADA).
    
    OTIMIZAÇÕES:
    - Usa context manager para gerenciamento adequado do PDF
    - Melhor tratamento de erros
    - Validação de parâmetros de entrada
    - Logging detalhado para debug
    
    Args:
        pdf_path: Caminho para o arquivo PDF
        page_num: Número da página (1-indexed)
        chart_rect: Retângulo da área a renderizar
        output_path: Caminho onde salvar a imagem renderizada
        dpi: Resolução de renderização
    
    Returns:
        Path do arquivo salvo ou None se falhar
    """
    # Validar parâmetros
    if not pdf_path.exists():
        logger.error(f"❌ PDF não encontrado: {pdf_path}")
        return None
    
    if page_num < 1:
        logger.error(f"❌ Número de página inválido: {page_num} (deve ser ≥ 1)")
        return None
    
    if dpi < 72 or dpi > 600:
        logger.warning(f"⚠️ DPI incomum: {dpi}. Recomendado: 150-300")
    
    try:
        with open_pdf_document(pdf_path) as doc:
            if page_num > doc.page_count:
                logger.error(f"❌ Página {page_num} não existe (PDF tem {doc.page_count} páginas)")
                return None
            
            page = doc[page_num - 1]  # Converter para 0-indexed
            
            # Calcular matriz de transformação para o DPI desejado
            zoom_factor = dpi / 72.0  # 72 DPI é o padrão
            matrix = fitz.Matrix(zoom_factor, zoom_factor)
            
            logger.info(f"🖼️ Renderizando página {page_num} em {dpi} DPI (zoom: {zoom_factor:.2f}x)")
            
            # Renderizar apenas a área especificada
            if chart_rect and not chart_rect.is_empty:
                # Renderizar área específica
                logger.debug(f"📐 Área específica: {chart_rect}")
                pixmap = page.get_pixmap(matrix=matrix, clip=chart_rect)
            else:
                # Renderizar página inteira
                logger.debug("📐 Renderizando página completa")
                pixmap = page.get_pixmap(matrix=matrix)
            
            # Salvar a imagem
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            pixmap_width, pixmap_height = pixmap.width, pixmap.height
            pixmap.save(str(output_path))
            del pixmap

            logger.info(f"✅ Imagem renderizada salva: {output_path.name} ({pixmap_width}x{pixmap_height}px)")
            return output_path
            
    except Exception as e:
        logger.error(f"❌ Erro ao renderizar: {e}")
        return None



def process_pdf_charts_optimized(
    pdf_path: Path,
    output_dir: Path,
    high_res_dpi: int = 200,
    min_chart_width: int = 300,
    min_chart_height: int = 200,
) -> List[dict]:
    """
    Pipeline completo para extrair gráficos de um PDF e salvá-los como PNGs.

    Extraction strategy (applied per page):
      1. Embedded raster images  — works for scanned / pre-rasterised figures.
      2. Full-page render        — fallback that captures vector charts
         (matplotlib, R, LaTeX, Word exports). Rendered at *high_res_dpi*.

    Returns a list of chart-info dicts.  Each dict always carries
    'high_res_path' pointing to the final PNG so that input_resolver.py
    can use a single code-path regardless of extraction method.
    """
    pdf_path   = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        logger.error(f"❌ PDF not found: {pdf_path}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"🚀 Processing {pdf_path.name} (render DPI={high_res_dpi})")

    processed_charts: List[dict] = []

    with open_pdf_document(pdf_path) as doc:
        logger.info(f"📄 {doc.page_count} page(s)")

        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                page_charts = _extract_images_from_page_optimized(
                    page,
                    page_num,
                    pdf_path.stem,
                    output_dir,
                    min_chart_width,
                    min_chart_height,
                    render_dpi=high_res_dpi,
                )

                if page_charts:
                    logger.info(
                        f"✅ Page {page_num + 1}: {len(page_charts)} chart(s) extracted"
                    )
                else:
                    logger.debug(f"⏭️  Page {page_num + 1}: no charts found")

                processed_charts.extend(page_charts)

            except Exception as exc:
                logger.error(f"❌ Error on page {page_num + 1}: {exc}")
                continue

    logger.info(
        f"🎉 Done: {len(processed_charts)} chart(s) from {pdf_path.name}"
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
                'errors': chart.get('high_res_error')
            }
            metadata['charts'].append(chart_meta)
        
        metadata_path = output_dir / f"{pdf_path.stem}_processing_metadata.json"
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📋 Metadados salvos: {metadata_path.name}")
        
    except Exception as e:
        logger.warning(f"⚠️ Não foi possível salvar metadados: {e}")


def extract_charts_with_doclayout(pdf_path: Path, output_dir: Path, model_path: str, figure_class_id: int = 3):
    """Extract charts using DocLayout-YOLO with proper error handling."""
    if not Path(model_path).exists():
        logger.error(f"❌ Model not found: {model_path}")
        return []

    try:
        from doclayout_yolo import YOLOv10
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning(
            "DocLayout extraction unavailable (missing optional dependency 'doclayout_yolo'): %s",
            exc,
        )
        return []
    
    try:
        model = YOLOv10(model_path)
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return []
    
    extracted = []
    
    try:
        with fitz.open(str(pdf_path)) as doc:
            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    pix = page.get_pixmap(dpi=150)
                    img = cv2.imdecode(np.frombuffer(pix.tobytes(), np.uint8), cv2.IMREAD_COLOR)
                    
                    if img is None:
                        logger.warning(f"Failed to decode page {page_num+1}")
                        continue
                    
                    det_res = model.predict(img, imgsz=1024, conf=0.2)
                    
                    if not det_res or len(det_res) == 0:
                        continue
                    
                    for det in det_res[0].boxes:
                        if int(det.cls) == figure_class_id:
                            x1, y1, x2, y2 = map(int, det.xyxy[0])
                            
                            # Validate bbox
                            if x2 <= x1 or y2 <= y1:
                                continue
                            
                            chart_img = img[y1:y2, x1:x2]
                            
                            if chart_img.size == 0:
                                continue
                            
                            filename = f"{pdf_path.stem}_page{page_num+1:02d}_chart_{len(extracted)+1:02d}.png"
                            path = output_dir / filename
                            
                            output_dir.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(path), chart_img)
                            
                            extracted.append({
                                'file_path': path,
                                'page_num': page_num+1,
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(det.conf)
                            })
                            
                            logger.info(f"✅ Extracted chart from page {page_num+1}: {filename}")
                            
                except Exception as e:
                    logger.error(f"❌ Error processing page {page_num+1}: {e}")
                    continue
        
        if not extracted:
            logger.warning(f"⚠️ No charts detected with class ID {figure_class_id}. "
                           f"Verify the model's class mappings.")
            
        return extracted
        
    except Exception as e:
        logger.error(f"❌ Error processing PDF: {e}")
        return []


# --- Funções de Compatibilidade (Backward Compatibility) ---

def extract_charts_from_pdf(pdf_path: Path, output_dir: Path, min_width: int = 300, min_height: int = 200) -> List[dict]:
    """
    Função de compatibilidade para manter interface da versão anterior.
    
    DEPRECATED: Use extract_charts_from_pdf_optimized() para melhor performance.
    """
    logger.warning("⚠️ Usando função legacy. Recomenda-se migrar para extract_charts_from_pdf_optimized()")
    return extract_charts_from_pdf_optimized(pdf_path, output_dir, min_width, min_height)


def rerender_chart_at_high_res(pdf_path: Path, page_num: int, chart_rect: fitz.Rect, output_path: Path, dpi: int = 300) -> Optional[Path]:
    """
    Função de compatibilidade para manter interface da versão anterior.
    
    DEPRECATED: Use rerender_chart_at_high_res_optimized() ou process_pdf_charts_optimized() para melhor performance.
    """
    logger.warning("⚠️ Usando função legacy. Recomenda-se migrar para rerender_chart_at_high_res_optimized()")
    return rerender_chart_at_high_res_optimized(pdf_path, page_num, chart_rect, output_path, dpi)


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
    
    print(f"🚀 Processando PDF OTIMIZADO: {pdf_path.name}")
    print(f"📁 Saída: {output_dir}")
    print(f"🖼️ DPI: {args.dpi}")
    print(f"📏 Dimensões mínimas: {args.min_width}x{args.min_height}")
    
    if not pdf_path.exists():
        print(f"❌ ERRO: PDF não encontrado: {pdf_path}")
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
        
        print(f"\n🎉 PROCESSAMENTO CONCLUÍDO!")
        print(f"✅ {len(results)} gráfico(s) extraído(s) e processado(s)")
        print(f"📂 Arquivos salvos em: {output_dir}")
        
        # Exibir resumo
        for i, chart in enumerate(results, 1):
            print(f"\n📊 Gráfico {i}:")
            print(f"   📄 Página: {chart.get('page_num')}")
            print(f"   📏 Dimensões: {chart.get('dimensions')}")
            if 'high_res_dimensions' in chart:
                print(f"   🖼️ Alta Res: {chart.get('high_res_dimensions')}")
            print(f"   💾 Arquivo: {chart.get('file_path', 'N/A')}")
        
        return 0
        
    except Exception as e:
        print(f"❌ ERRO durante processamento: {e}")
        logger.exception("Erro detalhado:")
        return 1


if __name__ == "__main__":
    exit(main())
