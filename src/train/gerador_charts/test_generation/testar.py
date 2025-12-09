

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import glob

CLASS_NAMES = {
    "CLASS_MAP_BAR": {
        0: "chart",
        1: "bar",
        2: "axis_title",
        3: "significance_marker",
        4: "error_bar",
        5: "legend",
        6: "chart_title",
        7: "data_label",
        8: "axis_labels"
    },
    "CLASS_MAP_PIE_OBJ": {
        0: "chart",
        1: "wedge",
        2: "legend",
        3: "chart_title",
        4: "data_label",
        5: "connector_line"
    },
    "CLASS_MAP_PIE_POSE": {
        0: "center_point",
        1: "arc_boundary",
        2: "wedge_center",
        3: "wedge_start",
        4: "wedge_end"
    },
    "CLASS_MAP_LINE_OBJ": {
        0: "chart",
        1: "line_segment",
        2: "axis_title",
        3: "legend",
        4: "chart_title",
        5: "data_label",
        6: "axis_labels"
    },
    "CLASS_MAP_LINE_POSE": {
        0: "line_boundary",
        1: "data_point",
        2: "inflection_point",
        3: "line_start",
        4: "line_end"
    },
    "CLASS_MAP_SCATTER": {
        0: "chart",
        1: "data_point",
        2: "axis_title",
        3: "significance_marker",
        4: "error_bar",
        5: "legend",
        6: "chart_title",
        7: "data_label",
        8: "axis_labels"
    },
    "CLASS_MAP_BOX": {
        0: "chart",
        1: "box",
        2: "axis_title",
        3: "significance_marker",
        4: "range_indicator",
        5: "legend",
        6: "chart_title",
        7: "median_line",
        8: "axis_labels",
        9: "outlier"
    },
    "CLASS_MAP_HISTOGRAM": {
        0: "chart",
        1: "bar",
        2: "axis_title",
        3: "legend",
        4: "chart_title",
        5: "data_label",
        6: "axis_labels"
    },
    "CLASS_MAP_HEATMAP": {
        0: "chart",
        1: "cell",
        2: "axis_title",
        3: "color_bar",
        4: "legend",
        5: "chart_title",
        6: "data_label",
        7: "axis_labels",
        8: "significance_marker"
    },
    "CLASS_MAP_AREA_POSE": {
        0: "area_fill",
        1: "area_boundary",
        2: "inflection_point",
        3: "area_start",
        4: "area_end"
    },
    "CLASS_MAP_AREA_OBJ": {
        0: "chart",
        1: "axis_title",
        2: "legend",
        3: "chart_title",
        4: "data_label",
        5: "axis_labels"
    }
}

# Color maps for each annotation type
COLOR_MAPS = {
    'bar': {
        1: (127, 127, 127),
        2: (34, 189, 188),
        3: (207, 190, 23),
        4: (120, 187, 255),
        5: (138, 223, 152),
        6: (232, 199, 174),
        7: (79, 79, 47),
        8: (200, 129, 51),
        9: (255, 129, 51)
    },
    'scatter': {
        1: (127, 127, 127),
        2: (34, 189, 188),
        3: (207, 190, 23),
        4: (120, 187, 255),
        5: (138, 223, 152),
        6: (232, 199, 174),
        7: (79, 79, 47),
        8: (200, 129, 51),
        9: (255, 129, 51)
    },
    'box': {
        0: (180, 119, 31),
        1: (44, 160, 44),
        2: (75, 86, 140),
        3: (194, 119, 227),
        4: (127, 127, 127),
        5: (34, 189, 188),
        6: (194, 119, 227),
        7: (255, 129, 51),
        8: (189, 103, 148),
        9: (75, 86, 140)
    },
    'histogram': {
        0: (180, 119, 31),
        1: (44, 160, 44),
        2: (75, 86, 140),
        3: (194, 119, 227),
        4: (127, 127, 127),
        5: (34, 189, 188),
        6: (194, 119, 227)
    },
    'heatmap': {
        0: (180, 119, 31),
        1: (44, 160, 44),
        2: (75, 86, 140),
        3: (194, 119, 227),
        4: (127, 127, 127),
        5: (34, 189, 188),
        6: (194, 119, 227),
        7: (127, 127, 127),
        8: (34, 189, 188)
    },
    'area_obj': {
        0: (180, 119, 31),
        1: (44, 160, 44),
        2: (75, 86, 140),
        3: (194, 119, 227),
        4: (127, 127, 127),
        5: (34, 189, 188)
    },
    'area_pose': {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 255, 0),
        4: (255, 0, 255)
    },
    'pie_obj': {
        0: (180, 119, 31),
        1: (14, 127, 255),
        2: (75, 86, 140),
        3: (194, 119, 227),
        4: (127, 127, 127),
        5: (40, 39, 214)
    },
    'pie_pose': {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 255, 0),
        4: (255, 0, 255)
    },
    'line_obj': {
        0: (180, 119, 31),
        1: (14, 127, 255),
        2: (44, 160, 44),
        3: (75, 86, 140),
        4: (194, 119, 227),
        5: (127, 127, 127),
        6: (34, 189, 188)
    },
    'line_pose': {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 255, 0),
        4: (255, 0, 255)
    }
}


def parse_pose_line(line):
    """
    Parse pose format annotation line.
    Format: <class_id> <x_center> <y_center> <width> <height> <kp1_x> <kp1_y> <kp2_x> <kp2_y> ...
    Returns: (class_id, bbox, keypoints)
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    
    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])
    
#     Extract keypoints (pairs of x, y coordinates)
    keypoints = []
    for i in range(5, len(parts), 2):
        if i + 1 < len(parts):
            kp_x = float(parts[i])
            kp_y = float(parts[i + 1])
            keypoints.append((kp_x, kp_y))
    
    return class_id, (x_center, y_center, width, height), keypoints


def find_label_file(image_path, base_dir=None):
    """
    Intelligently find label file for given image.
    Searches in multiple possible locations:
    1. Same directory as image
    2. labels/ directory (standard)
    3. Dual-format directories (area_obj_labels, area_pose_labels, etc.)
    Returns: (txt_path, annotation_type) or (None, None) if not found
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
#     Determine base directory
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(image_path))
    
#     Search locations in priority order
    search_locations = [
        (os.path.dirname(image_path), 'standard'),
        (os.path.join(base_dir, 'labels'), 'standard'),
        (os.path.join(base_dir, 'area_obj_labels'), 'area_obj'),
        (os.path.join(base_dir, 'area_pose_labels'), 'area_pose'),
        (os.path.join(base_dir, 'pie_obj_labels'), 'pie_obj'),
        (os.path.join(base_dir, 'pie_pose_labels'), 'pie_pose'),
        (os.path.join(base_dir, 'line_obj_labels'), 'line_obj'),
        (os.path.join(base_dir, 'line_pose_labels'), 'line_pose')
    ]
    
    for label_dir, annotation_type in search_locations:
        txt_path = os.path.join(label_dir, f"{base_name}.txt")
        if os.path.exists(txt_path):
            return txt_path, annotation_type
    
    return None, None


def detect_annotation_format(txt_path):
    """
    Detect annotation format: 'yolo' (5 values) or 'pose' (5+ values with keypoints).
    Also detect annotation type from directory structure.
    Returns: (format_type, annotation_type)
    """
    normalized_path = txt_path.replace('\\', '/').lower()
    
    annotation_type = 'standard'
    if 'area_obj_labels' in normalized_path:
        annotation_type = 'area_obj'
    elif 'area_pose_labels' in normalized_path:
        annotation_type = 'area_pose'
    elif 'pie_obj_labels' in normalized_path:
        annotation_type = 'pie_obj'
    elif 'pie_pose_labels' in normalized_path:
        annotation_type = 'pie_pose'
    elif 'line_obj_labels' in normalized_path:
        annotation_type = 'line_obj'
    elif 'line_pose_labels' in normalized_path:
        annotation_type = 'line_pose'
    
    format_type = 'yolo'
    try:
        with open(txt_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                parts = first_line.split()
                if len(parts) > 5:
                    format_type = 'pose'
    except:
        pass
    
    return format_type, annotation_type


def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    """Converte coordenadas YOLO normalizadas para bbox em pixels."""
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return x1, y1, x2, y2


def read_yolo_annotations(txt_path):
    """
    Read annotations from file. Supports both YOLO and pose formats.
    Returns: (annotations, format_type, annotation_type)
    """
    annotations = []
    format_type, annotation_type = detect_annotation_format(txt_path)
    
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if format_type == 'pose':
                    parsed = parse_pose_line(line)
                    if parsed:
                        class_id, bbox, keypoints = parsed
                        annotations.append({
                            'class_id': class_id,
                            'bbox': bbox,
                            'keypoints': keypoints,
                            'format': 'pose'
                        })
                else:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        annotations.append({
                            'class_id': class_id,
                            'bbox': (x_center, y_center, width, height),
                            'keypoints': [],
                            'format': 'yolo'
                        })
    except FileNotFoundError:
        print(f"⚠ Arquivo de anotação não encontrado: {txt_path}")
        return [], format_type, annotation_type
    
    return annotations, format_type, annotation_type


def draw_keypoints_opencv(img, keypoints, img_w, img_h, color, radius=3):
    """Draw keypoints on OpenCV image."""
    for kp_x, kp_y in keypoints:
        px = int(kp_x * img_w)
        py = int(kp_y * img_h)
        px = max(0, min(px, img_w - 1))
        py = max(0, min(py, img_h - 1))
        cv2.circle(img, (px, py), radius, color, -1)
        cv2.circle(img, (px, py), radius + 1, (255, 255, 255), 1)
    
    if len(keypoints) > 1:
        for i in range(len(keypoints) - 1):
            px1 = int(keypoints[i][0] * img_w)
            py1 = int(keypoints[i][1] * img_h)
            px2 = int(keypoints[i + 1][0] * img_w)
            py2 = int(keypoints[i + 1][1] * img_h)
            cv2.line(img, (px1, py1), (px2, py2), color, 1)


def draw_keypoints_matplotlib(ax, keypoints, img_w, img_h, color):
    """Draw keypoints on matplotlib axis."""
    for kp_x, kp_y in keypoints:
        px = kp_x * img_w
        py = kp_y * img_h
        circle = patches.Circle((px, py), radius=5,
                               edgecolor='white', facecolor=color,
                               linewidth=1.5, zorder=10)
        ax.add_patch(circle)
    
    if len(keypoints) > 1:
        xs = [kp[0] * img_w for kp in keypoints]
        ys = [kp[1] * img_h for kp in keypoints]
        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.7, zorder=9)


def visualize_with_opencv(image_path, annotations, format_type, annotation_type,
                         save_path=None, show=True):
    """Visualiza anotações usando OpenCV."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Erro ao carregar imagem: {image_path}")
        return
    
    img_h, img_w = img.shape[:2]
    
    class_map = CLASS_NAMES.get(annotation_type, CLASS_NAMES)
    color_map = COLOR_MAPS.get(annotation_type, COLOR_MAPS)
    
    for ann in annotations:
        class_id = ann['class_id']
        x_center, y_center, width, height = ann['bbox']
        keypoints = ann.get('keypoints', [])
        
        x1, y1, x2, y2 = yolo_to_bbox(x_center, y_center, width, height, img_w, img_h)
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))
        
        color = color_map.get(class_id, (128, 128, 128))
        class_name = class_map.get(class_id, f"Class_{class_id}")
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        if keypoints:
            draw_keypoints_opencv(img, keypoints, img_w, img_h, color, radius=4)
        
        label = f"{class_name} ({class_id})"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 10, label_size[1])
        cv2.rectangle(img, (x1, label_y - label_size[1]),
                     (x1 + label_size[0], label_y + 2), color, -1)
        cv2.putText(img, label, (x1, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    legend_y = 30
    for class_id, name in class_map.items():
        color = color_map.get(class_id, (128, 128, 128))
        legend_text = f"{class_id}: {name}"
        cv2.rectangle(img, (10, legend_y - 20), (30, legend_y), color, -1)
        cv2.putText(img, legend_text, (35, legend_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        legend_y += 25
    
    format_label = f"Format: {format_type.upper()} | Type: {annotation_type}"
    cv2.putText(img, format_label, (10, img_h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, format_label, (10, img_h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    if save_path:
        cv2.imwrite(save_path, img)
        print(f"✅ Visualização salva: {save_path}")
    
    if show:
        cv2.imshow('YOLO Annotations', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return img


def visualize_with_matplotlib(image_path, annotations, format_type, annotation_type,
                              save_path=None, show=True):
    """Visualiza anotações usando matplotlib."""
    img = Image.open(image_path)
    img_w, img_h = img.size
    
    class_map = CLASS_NAMES.get(annotation_type, CLASS_NAMES)
    color_map = COLOR_MAPS.get(annotation_type, COLOR_MAPS)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    
    for ann in annotations:
        class_id = ann['class_id']
        x_center, y_center, width, height = ann['bbox']
        keypoints = ann.get('keypoints', [])
        
        x1, y1, x2, y2 = yolo_to_bbox(x_center, y_center, width, height, img_w, img_h)
        
        color = np.array(color_map.get(class_id, (128, 128, 128))[::-1]) / 255.0
        class_name = class_map.get(class_id, f"Class_{class_id}")
        
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        if keypoints:
            draw_keypoints_matplotlib(ax, keypoints, img_w, img_h, color)
        
        ax.text(x1, y1-5, f"{class_name} ({class_id})",
               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
               fontsize=8, color='white', weight='bold')
    
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.axis('off')
    plt.title(f"Anotações - {os.path.basename(image_path)} | {format_type.upper()} | {annotation_type}",
             fontsize=14, weight='bold')
    
    legend_elements = [patches.Patch(facecolor=np.array(color_map[cid][::-1])/255.0,
                                    edgecolor='black', label=f"{cid}: {name}")
                      for cid, name in class_map.items()]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualização salva: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def process_single_image(image_path, output_dir=None, method='matplotlib', show=True, base_dir=None):
    """Processa uma única imagem."""
    txt_path, annotation_type_hint = find_label_file(image_path, base_dir)
    
    if txt_path is None:
        print(f"⚠ Arquivo de anotação não encontrado para: {image_path}")
        return
    
    if output_dir is None:
        print(f'Imagem {image_path} anotada, salva no diretório output')
        output_dir = 'output/'
    
    annotations, format_type, annotation_type = read_yolo_annotations(txt_path)
    
    if annotation_type == 'standard' and annotation_type_hint != 'standard':
        annotation_type = annotation_type_hint
    
    if not annotations:
        print(f"⚠ Nenhuma anotação encontrada em: {txt_path}")
        return
    
    print(f"📊 Processando: {os.path.basename(image_path)}")
    print(f"  📄 Label: {os.path.basename(txt_path)}")
    print(f"  Format: {format_type} | Type: {annotation_type} | {len(annotations)} anotações")
    
    save_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(output_dir, f"{basename}_annotated_{annotation_type}.png")
    
    if method == 'opencv':
        visualize_with_opencv(image_path, annotations, format_type, annotation_type, save_path, show)
    else:
        visualize_with_matplotlib(image_path, annotations, format_type, annotation_type, save_path, show)


def process_directory(input_dir, output_dir=None, method='matplotlib', show_each=False):
    """Processa todas as imagens em um diretório e seus subdiretórios de labels."""
    images_dir = input_dir
    if os.path.basename(input_dir) != 'images':
        potential_images_dir = os.path.join(input_dir, 'images')
        if os.path.exists(potential_images_dir):
            images_dir = potential_images_dir
    
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, f'*{ext}')))
    
    if not image_files:
        print(f"❌ Nenhuma imagem encontrada em {images_dir}")
        return
    
    print(f"✅ Encontradas {len(image_files)} imagens em {images_dir}")
    
    annotation_groups = {
        'bar': [],
        'box': [],
        'scatter': [],
        'heatmap': [],
        'histogram': [],
        'area_obj': [],
        'area_pose': [],
        'pie_obj': [],
        'pie_pose': [],
        'line_obj': [],
        'line_pose': []
    }
    
    base_dir = os.path.dirname(images_dir) if os.path.basename(images_dir) == 'images' else input_dir
    
    for image_path in sorted(image_files):
        txt_path, annotation_type = find_label_file(image_path, base_dir)
        if txt_path:
            annotation_groups[annotation_type].append((image_path, txt_path))
    
    print("\n📊 Anotações encontradas:")
    for ann_type, file_list in annotation_groups.items():
        if file_list:
            print(f"  {ann_type}: {len(file_list)} imagens")
    print()
    
    processed = 0
    for ann_type, file_list in annotation_groups.items():
        if not file_list:
            continue
        
        print(f"🔄 Processando {len(file_list)} imagens tipo '{ann_type}'...")
        
        for image_path, txt_path in file_list:
            try:
                annotations, format_type, detected_type = read_yolo_annotations(txt_path)
                if not annotations:
                    print(f"⚠ Nenhuma anotação em: {os.path.basename(txt_path)}")
                    continue
                
                print(f"  📊 {os.path.basename(image_path)} - {format_type.upper()} - {len(annotations)} anotações")
                
                save_path = None
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    basename = os.path.splitext(os.path.basename(image_path))[0]
                    save_path = os.path.join(output_dir, f"{basename}_annotated_{ann_type}.png")
                
                if method == 'opencv':
                    visualize_with_opencv(image_path, annotations, format_type, ann_type, save_path, show_each)
                else:
                    visualize_with_matplotlib(image_path, annotations, format_type, ann_type, save_path, show_each)
                
                processed += 1
            except Exception as e:
                print(f"❌ Erro processando {image_path}: {e}")
    
    print(f"\n✅ Processadas {processed}/{len(image_files)} imagens")


def main():
    parser = argparse.ArgumentParser(description='Visualizador de anotações YOLO para gráficos (suporta dual-format)')
    parser.add_argument('input', help='Caminho para imagem, diretório ou output_dir')
    parser.add_argument('-o', '--output', help='Diretório de saída para imagens anotadas')
    parser.add_argument('-m', '--method', choices=['matplotlib', 'opencv'], default='matplotlib',
                       help='Método de visualização (default: matplotlib)')
    parser.add_argument('--show', action='store_true',
                       help='Mostrar cada imagem (útil para diretórios)')
    parser.add_argument('--no-show', action='store_true',
                       help='Não mostrar imagens, apenas salvar')
    parser.add_argument('--base-dir', help='Diretório base para busca de labels')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Caminho não encontrado: {args.input}")
        return
    
    show = (not args.no_show) if args.no_show else args.show if os.path.isdir(args.input) else True
    
    print("=" * 60)
    print("Visualizador de Anotações YOLO (Dual-Format Support)")
    print("=" * 60)
    print(f"📂 Input: {args.input}")
    if args.output:
        print(f"💾 Output: {args.output}")
    print(f"🎨 Método: {args.method}")
    print("=" * 60)
    
    if os.path.isfile(args.input):
        process_single_image(args.input, args.output, args.method, show, args.base_dir)
    else:
        process_directory(args.input, args.output, args.method, show)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        print("=" * 60)
        print("Visualizador de Anotações YOLO - Modo Interativo")
        print("=" * 60)
        
        default_dir = 'test'
        if os.path.exists(default_dir):
            print(f"📂 Usando diretório padrão: {default_dir}")
            process_directory(default_dir, output_dir='visualized_annotations', show_each=False)
        else:
            print("\n📖 Uso:")
            print("  python testar.py <image_or_dir>")
            print("  python testar.py test")
            print("  python testar.py test -o visualized --show")
            print("\n💡 Para visualizar dual-format annotations:")
            print("  python testar.py test/area_pose_labels")
            print("  python testar.py test/pie_obj_labels")
    else:
        main()


