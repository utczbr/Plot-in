"""
Script para visualizar as deteções do modelo detect_histogram.onnx.
Desenha todas as caixas de deteção com cores diferentes para cada classe (0-6).
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Adicionar o diretório src ao path para imports
sys.path.insert(0, str(Path(__file__).parent))

from core.class_maps import CLASS_MAP_HISTOGRAM
from utils.inference import preprocess_with_letterbox, run_inference_on_image


# Cores distintas para cada classe (BGR - OpenCV usa formato BGR)
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 255),       # Vermelho - chart
    1: (0, 255, 0),       # Verde - bar
    2: (255, 0, 0),       # Azul - axis_title
    3: (255, 255, 0),     # Ciano - legend
    4: (255, 0, 255),     # Magenta - chart_title
    5: (0, 255, 255),     # Amarelo - data_label
    6: (0, 165, 255),     # Laranja - axis_labels
}


def draw_detections(
    img: np.ndarray,
    detections: List[Dict],
    class_map: Dict[int, str],
    colors: Dict[int, Tuple[int, int, int]],
    font_scale: float = 0.6,
    thickness: int = 2,
) -> np.ndarray:
    """
    Desenha as caixas de deteção na imagem.

    Args:
        img: Imagem original (numpy array BGR)
        detections: Lista de deteções com chaves 'xyxy', 'conf', 'cls'
        class_map: Mapeamento de ID de classe para nome
        colors: Mapeamento de ID de classe para cor (BGR)
        font_scale: Escala da fonte
        thickness: Espessura da linha

    Returns:
        Imagem anotada com as caixas de deteção
    """
    annotated = img.copy()

    for det in detections:
        if 'xyxy' not in det:
            continue

        cls_id = int(det['cls'])
        conf = float(det['conf'])
        x1, y1, x2, y2 = map(int, det['xyxy'])

        # Obter cor e nome da classe
        color = colors.get(cls_id, (128, 128, 128))
        class_name = class_map.get(cls_id, f'class_{cls_id}')

        # Desenhar retângulo
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Preparar label com classe e confiança
        label = f"{class_name}: {conf:.2f}"

        # Calcular tamanho do texto para fundo
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Desenhar fundo para o texto
        cv2.rectangle(
            annotated,
            (x1, y1 - text_h - baseline - 5),
            (x1 + text_w, y1),
            color,
            -1,  # Preenchido
        )

        # Desenhar texto
        cv2.putText(
            annotated,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # Texto branco
            thickness,
        )

    return annotated


def visualize_histogram_model(
    model_path: Path,
    image_dir: Path,
    output_dir: Path,
    conf_threshold: float = 0.5,
    input_size: Tuple[int, int] = (640, 640),
) -> None:
    """
    Processa todas as imagens de histograma e desenha as deteções.

    Args:
        model_path: Caminho para o modelo ONNX
        image_dir: Diretório com as imagens de teste
        output_dir: Diretório para salvar as imagens anotadas
        conf_threshold: Threshold de confiança para filtrar deteções
        input_size: Tamanho de entrada do modelo (largura, altura)
    """
    # Criar diretório de saída se não existir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Carregar modelo
    print(f"Carregando modelo: {model_path}")
    session = ort.InferenceSession(str(model_path))

    # Obter todas as imagens
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(image_dir.glob(ext))

    if not image_paths:
        print(f"Nenhuma imagem encontrada em: {image_dir}")
        return

    print(f"Encontradas {len(image_paths)} imagens para processar")
    print(f"Classes do modelo: {CLASS_MAP_HISTOGRAM}")
    print("-" * 60)

    # Processar cada imagem
    for img_path in sorted(image_paths):
        print(f"\nProcessando: {img_path.name}")

        # Carregar imagem
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Erro ao carregar imagem: {img_path}")
            continue

        print(f"  Dimensões: {img.shape}")

        # Executar inferência
        detections = run_inference_on_image(
            session=session,
            img=img,
            conf_threshold=conf_threshold,
            class_map=CLASS_MAP_HISTOGRAM,
            input_size=input_size,
            nms_threshold=0.45,
            model_output_type="bbox",
        )

        print(f"  Deteções encontradas: {len(detections)}")

        # Agrupar por classe para relatório
        class_counts: Dict[int, int] = {}
        for det in detections:
            cls_id = int(det['cls'])
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

        for cls_id, count in sorted(class_counts.items()):
            class_name = CLASS_MAP_HISTOGRAM.get(cls_id, f"class_{cls_id}")
            print(f"    Classe {cls_id} ({class_name}): {count} deteções")

        # Desenhar deteções na imagem
        # Calcular font_scale e thickness baseados no tamanho da imagem
        h, w = img.shape[:2]
        dynamic_scale = (w + h) / 2500
        font_scale = max(0.5, dynamic_scale)
        thickness = max(1, int(dynamic_scale * 2))

        annotated = draw_detections(
            img=img,
            detections=detections,
            class_map=CLASS_MAP_HISTOGRAM,
            colors=CLASS_COLORS,
            font_scale=font_scale,
            thickness=thickness,
        )

        # Salvar imagem anotada
        output_path = output_dir / f"annotated_{img_path.name}"
        cv2.imwrite(str(output_path), annotated)
        print(f"  Salvou: {output_path}")

    print("\n" + "=" * 60)
    print("Processamento concluído!")
    print(f"Imagens anotadas salvas em: {output_dir}")


def main():
    """Ponto de entrada principal."""
    # Definir caminhos
    base_dir = Path(__file__).parent
    model_path = base_dir / "models" / "detect_histogram.onnx"
    image_dir = base_dir / "images" / "histogram"
    output_dir = base_dir / "images" / "histogram" / "annotated"

    # Verificar se o modelo existe
    if not model_path.exists():
        print(f"Erro: Modelo não encontrado em: {model_path}")
        sys.exit(1)

    # Verificar se o diretório de imagens existe
    if not image_dir.exists():
        print(f"Erro: Diretório de imagens não encontrado em: {image_dir}")
        sys.exit(1)

    # Executar visualização
    visualize_histogram_model(
        model_path=model_path,
        image_dir=image_dir,
        output_dir=output_dir,
        conf_threshold=0.5,
    )


if __name__ == "__main__":
    main()
