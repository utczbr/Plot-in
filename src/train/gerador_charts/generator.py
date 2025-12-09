from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import List, Dict, Tuple
BoundingBox = namedtuple('BoundingBox', ['x0', 'y0', 'x1', 'y1'])
import time

def validate_coordinates(coords, context="unknown"):
    """
    Validate coordinate lists for debugging and consistency.
    
    Args:
        coords: List of coordinate tuples (x, y, [visibility])
        context: Context string for debugging output
    
    Returns:
        Boolean indicating if coordinates are valid
    """
    if not coords:
        if GENERATION_CONFIG.get('debug_coords', False):
            print(f"DEBUG [COORD-VALIDATION] {context}: No coordinates provided")
        return True  # Empty is valid
    
    try:
        for i, coord in enumerate(coords):
            if len(coord) < 2:
                if GENERATION_CONFIG.get('debug_coords', False):
                    print(f"DEBUG [COORD-VALIDATION] {context}: Invalid coord {i}, too few elements: {coord}")
                continue
            
            x, y = coord[0], coord[1]
            if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                if GENERATION_CONFIG.get('debug_coords', False):
                    print(f"DEBUG [COORD-VALIDATION] {context}: Invalid coordinate type at {i}: x={x}, y={y}")
                continue
            
            # Check for inf/nan values
            if not (np.isfinite(x) and np.isfinite(y)):
                if GENERATION_CONFIG.get('debug_coords', False):
                    print(f"DEBUG [COORD-VALIDATION] {context}: Non-finite coordinate at {i}: x={x}, y={y}")
                continue
            
            if GENERATION_CONFIG.get('debug_coords', False):
                if len(coord) >= 3:  # Has visibility
                    print(f"DEBUG [COORD-VALIDATION] {context}: Valid coord {i}: ({x:.2f}, {y:.2f}, vis={coord[2]})")
                else:
                    print(f"DEBUG [COORD-VALIDATION] {context}: Valid coord {i}: ({x:.2f}, {y:.2f})")
        
        return True
    except Exception as e:
        if GENERATION_CONFIG.get('debug_coords', False):
            print(f"DEBUG [COORD-VALIDATION] {context}: Error validating coordinates: {e}")
        return False


def verify_pose_format(annotations: List[Dict], context="unknown"):
    """
    Verify pose annotation format and content for debugging.
    
    Args:
        annotations: List of pose annotation dictionaries
        context: Context string for debugging output
    
    Returns:
        Boolean indicating if format is valid
    """
    if GENERATION_CONFIG.get('debug_coords', False):
        print(f"DEBUG [POSE-VERIFICATION] {context}: Checking {len(annotations)} annotations")
    
    all_valid = True
    
    for i, ann in enumerate(annotations):
        if 'class_id' not in ann:
            if GENERATION_CONFIG.get('debug_coords', False):
                print(f"DEBUG [POSE-VERIFICATION] {context}: Annotation {i} missing class_id")
            all_valid = False
            continue
            
        if 'bbox' not in ann:
            if GENERATION_CONFIG.get('debug_coords', False):
                print(f"DEBUG [POSE-VERIFICATION] {context}: Annotation {i} missing bbox")
            all_valid = False
            continue
            
        if 'keypoints' not in ann:
            if GENERATION_CONFIG.get('debug_coords', False):
                print(f"DEBUG [POSE-VERIFICATION] {context}: Annotation {i} missing keypoints")
            all_valid = False
            continue
        
        bbox = ann['bbox']
        keypoints = ann['keypoints']
        
        # Verify bbox format (4 values: center_x, center_y, width, height)
        if len(bbox) != 4:
            if GENERATION_CONFIG.get('debug_coords', False):
                print(f"DEBUG [POSE-VERIFICATION] {context}: Annotation {i} bbox has {len(bbox)} elements, expected 4")
            all_valid = False
            continue
        
        center_x, center_y, width, height = bbox
        if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0):
            if GENERATION_CONFIG.get('debug_coords', False):
                print(f"DEBUG [POSE-VERIFICATION] {context}: Annotation {i} bbox center not normalized [0,1]: ({center_x:.3f}, {center_y:.3f})")
        
        if not (0.0 <= width <= 1.0 and 0.0 <= height <= 1.0):
            if GENERATION_CONFIG.get('debug_coords', False):
                print(f"DEBUG [POSE-VERIFICATION] {context}: Annotation {i} bbox size not normalized [0,1]: ({width:.3f}, {height:.3f})")
        
        # Verify keypoints format (51 keypoints for line/area, 5 for pie)
        expected_kpts = 51  # For line and area charts
        if len(keypoints) != expected_kpts:
            # --- INÍCIO DA MODIFICAÇÃO ---
            # Handle pie chart case with 5 keypoints
            if len(keypoints) == 5:
                # For pie charts, 5 keypoints is acceptable
                if GENERATION_CONFIG.get('debug_coords', False):
                    print(f"DEBUG [POSE-VERIFICATION] {context}: Annotation {i} has {len(keypoints)} keypoints (pie chart - acceptable)")
            # --- FIM DA MODIFICAÇÃO ---
            else:
                if GENERATION_CONFIG.get('debug_coords', False):
                    print(f"DEBUG [POSE-VERIFICATION] {context}: Annotation {i} has {len(keypoints)} keypoints, expected {expected_kpts} or 5")
                all_valid = False
                continue
        
        # Check each keypoint (x, y, visibility)
        visible_count = 0
        invalid_coords = 0
        for j, kpt in enumerate(keypoints):
            if len(kpt) != 3:
                if GENERATION_CONFIG.get('debug_coords', False):
                    print(f"DEBUG [POSE-VERIFICATION] {context}: Annotation {i} keypoint {j} has {len(kpt)} elements, expected 3 (x, y, vis)")
                invalid_coords += 1
                all_valid = False
                continue
            
            x, y, vis = kpt
            
            # Check normalization for visible keypoints
            if vis > 0:
                visible_count += 1
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    if GENERATION_CONFIG.get('debug_coords', False):
                        print(f"DEBUG [POSE-VERIFICATION] {context}: Annotation {i} keypoint {j} not normalized [0,1]: ({x:.3f}, {y:.3f}, vis={vis})")
                    invalid_coords += 1
                    all_valid = False
                    continue
            
            # Check visibility value
            if vis not in [0, 2]:
                if GENERATION_CONFIG.get('debug_coords', False):
                    print(f"DEBUG [POSE-VERIFICATION] {context}: Annotation {i} keypoint {j} has invalid visibility: {vis} (expected 0 or 2)")
                all_valid = False
                continue
        
        # CRITICAL: Add monotonicity check for x coordinates (ensure non-decreasing order)
        # Only for 51-point annotations (line/area charts), not for 3-point pie annotations
        if len(keypoints) == 51:
            x_coords_visible = [kpt[0] for kpt in keypoints if kpt[2] > 0]  # Only visible keypoints
            if len(x_coords_visible) > 1:
                is_monotonic = all(x_coords_visible[i] <= x_coords_visible[i+1] 
                                 for i in range(len(x_coords_visible)-1))
                if not is_monotonic:
                    if GENERATION_CONFIG.get('debug_coords', False):
                        print(f"DEBUG [POSE-VERIFICATION] {context}: Annotation {i} x coordinates not monotonically non-decreasing")
                    # This may not be an error if axes are inverted, so just warn for debugging
                    if GENERATION_CONFIG.get('debug_coords', False):
                        print(f"DEBUG [POSE-VERIFICATION] {context}: Annotation {i} x coordinates: {x_coords_visible[:5]}... (first 5)")
        
        if GENERATION_CONFIG.get('debug_coords', False) and invalid_coords == 0:
            print(f"DEBUG [POSE-VERIFICATION] {context}: Annotation {i} format OK ({visible_count}/{len(keypoints)} visible)")
    
    return all_valid
import matplotlib
matplotlib.use('Agg')

from matplotlib import patches, rcParams, transforms, colormaps
import matplotlib.pyplot as plt
import numpy as np
import os, io, random, math, traceback, warnings, json, subprocess
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw, ImageFont
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from matplotlib.container import ErrorbarContainer
import matplotlib.lines
from matplotlib.collections import PolyCollection, PathCollection, QuadMesh

from themes import THEMES, CHART_TITLES, SCIENTIFIC_Y_LABELS, BUSINESS_Y_LABELS, SCIENTIFIC_X_LABELS, BUSINESS_X_LABELS
from effects import (
    apply_jpeg_compression_effect, apply_noise_effect, apply_blur_effect, 
    apply_motion_blur_effect, apply_low_res_effect, apply_pixelation_effect, 
    apply_posterize_effect, apply_color_variation_effect, apply_ui_chrome_effect, 
    apply_watermark_effect, apply_vignette_effect, apply_scanner_streaks_effect, 
    apply_clipping_effect, apply_printing_artifacts_effect, apply_mouse_cursor_effect, 
    apply_text_degradation_effect, apply_grid_occlusion_effect, apply_scan_rotation_effect, 
    apply_grayscale_effect, apply_perspective_effect
)

from chart import (_generate_bar_chart, _generate_line_chart, _generate_scatter_chart, 
                   _generate_boxplot_chart, _generate_heatmap_chart, _generate_pie_chart, 
                   _generate_area_chart, _generate_histogram, add_data_labels, apply_chart_theme)


warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# FIXED KEYPOINT DEFINITIONS FOR YOLO POSE COMPLIANCE
# ============================================================================

@dataclass
class KeypointConfig:
    """Fixed keypoint configuration for YOLO pose."""
    num_keypoints: int
    skeleton: List[Tuple[int, int]]  # Keypoint connections
    keypoint_names: List[str]

# Fixed Keypoint Definitions (CRITICAL for YOLO Compliance)
LINE_KEYPOINT_CONFIG = KeypointConfig(
    num_keypoints=51,  # Fixed: 1 start + 25 boundary + 20 inflections + 4 extrema + 1 end
    keypoint_names=[
        'start',           # 0
        *[f'boundary_{i}' for i in range(25)],  # 1-25: Fixed 25 boundary points
        *[f'inflection_{i}' for i in range(20)],  # 26-45: Up to 20 inflections (pad if less)
        'peak_1', 'peak_2', 'valley_1', 'valley_2',  # 46-49: Up to 2 peaks/valleys
        'end'  # 50
    ],
    skeleton=[
        (0, 1), (1, 2), *[(i, i+1) for i in range(1, 25)], (25, 50)  # Connect boundary points
    ]
)

AREA_KEYPOINT_CONFIG = KeypointConfig(
    num_keypoints=51,  # 1 start + 25 top boundary + 24 bottom boundary + 1 end
    keypoint_names=[
        'start',
        *[f'top_{i}' for i in range(25)],     # 1-25: Top boundary
        *[f'bottom_{i}' for i in range(24)],  # 26-49: Bottom boundary
        'end'
    ],
    skeleton=[
        (0, 1), *[(i, i+1) for i in range(1, 25)], (25, 50),  # Top
        (0, 26), *[(i, i+1) for i in range(26, 49)], (49, 50)  # Bottom
    ]
)

PIE_KEYPOINT_CONFIG = KeypointConfig(
    num_keypoints=17,  # 1 center + 1 wedge_center + 15 arc points (enough for 360°/15=24° resolution)
    keypoint_names=[
        'pie_center',          # 0
        'wedge_center',        # 1
        *[f'arc_{i}' for i in range(15)]  # 2-16: Arc boundary
    ],
    skeleton=[
        (0, 1),  # Center to wedge center
        (1, 2), *[(i, i+1) for i in range(2, 16)], (16, 2)  # Arc closed loop
    ]
)

from scipy.interpolate import interp1d
import numpy as np

def _curvature_weights(poly):
    # poly: list of (x, y, idx) in order
    n = len(poly)
    if n < 3:
        return np.ones(max(n-1, 1), dtype=float)
    pts = np.array([[p[0], p[1]] for p in poly], dtype=float)
    # segment lengths
    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1) + 1e-8
    # turn angles (curvature proxy) at interior vertices
    v1 = seg[:-1] / seg_len[:-1, None]
    v2 = seg[1:] / seg_len[1:, None]
    cosang = np.clip((v1 * v2).sum(axis=1), -1.0, 1.0)
    ang = np.arccos(cosang)  # radians
    # expand angles to segment weights: assign angle to adjacent segments
    kappa = np.zeros_like(seg_len)
    if len(ang) > 0:
        kappa[:-1] += ang
        kappa[1:]  += ang
    # final weight per segment: small base + curvature emphasis
    w = 1e-3 + kappa
    return w

def _collect_anchors_from_series(serieskpts, max_anchors=10):
    # serieskpts fields contain tuples (x, y, idx); use only idx to avoid duplicates
    anchors = set()
    for key in ("peaks", "valleys", "inflections"):
        pts = serieskpts.get(key, []) or []
        for x, y, i in pts:
            anchors.add(int(i))
            if len(anchors) >= max_anchors:
                break
    # Always include endpoints if available within allpoints
    allpts = serieskpts.get("all_points", []) or serieskpts.get("fill_top", []) or serieskpts.get("boundary_points", []) or []
    if allpts:
        anchors.add(int(allpts[0][2]))
        anchors.add(int(allpts[-1][2]))
    return sorted(anchors)

def resample_keypoints_adaptive(points, target=51, anchors_idx=None):
    """
    points: list[(x, y, idx)] in path order
    anchors_idx: sorted list of original indices that must appear in output
    Returns: list[(x, y, idx)] length==target
    """
    if not points:
        return [(0.0, 0.0, 0)] * target
    if len(points) == 1:
        x, y, i = points[0]
        return [(float(x), float(y), int(i))] * target

    # Ensure order by original idx if present
    pts = sorted(points, key=lambda p: p[2])
    # Build weighted arc-length parameterization
    seg_w = _curvature_weights(pts)
    seg_len = np.linalg.norm(np.diff(np.array([[p[0], p[1]] for p in pts], float), axis=0), axis=1) + 1e-8
    wlen = seg_w * seg_len
    t = np.concatenate([[0.0], np.cumsum(wlen)])
    T = t[-1] if t[-1] > 0 else float(len(pts) - 1)

    # Anchor preservation: map anchor idx to t, and seed sample positions with them
    anchor_ts = []
    if anchors_idx:
        idx2t = {}
        for k in range(len(pts)-1):
            idx2t[pts[k][2]] = t[k]
        idx2t[pts[-1][2]] = t[-1]
        for ai in anchors_idx:
            if ai in idx2t:
                anchor_ts.append(idx2t[ai])
    anchor_ts = sorted(set(anchor_ts))

    # Distribute remaining positions between anchors proportionally to weighted length
    m = target
    base_ts = []
    if anchor_ts:
        # Ensure first/last
        if anchor_ts[0] > 0.0:
            anchor_ts = [0.0] + anchor_ts
        if anchor_ts[-1] < T:
            anchor_ts = anchor_ts + [T]
        # Allocate samples per span
        remaining = m - len(anchor_ts)
        if remaining < 0:
            # Too many anchors; downsample anchors uniformly
            sel = np.linspace(0, len(anchor_ts)-1, m).astype(int)
            base_ts = [anchor_ts[s] for s in sel]
        else:
            base_ts = list(anchor_ts)
            # per-span quota
            span_lengths = np.diff(anchor_ts)
            total_span = span_lengths.sum() if span_lengths.sum() > 0 else 1.0
            quotas = np.floor(remaining * (span_lengths / total_span)).astype(int)
            # fix rounding
            while quotas.sum() < remaining:
                quotas[np.argmax(span_lengths - quotas)] += 1
            # fill spans
            for s, q in enumerate(quotas):
                if q <= 0:
                    continue
                a, b = anchor_ts[s], anchor_ts[s+1]
                for r in range(1, q+1):
                    base_ts.append(a + r * (b - a) / (q + 1))
    else:
        # No anchors: weighted-uniform along [0, T]
        base_ts = list(np.linspace(0.0, T, m))

    # Interpolate along weighted t to get (x, y, idx)
    xy = np.array([[p[0], p[1]] for p in pts], float)
    orig_idx = np.array([p[2] for p in pts], int)
    # Build piecewise-linear interpolation in t
    out = []
    base_ts = np.array(sorted(base_ts))
    # Map t to segment k with t[k] <= tau <= t[k+1]
    k = 0
    for tau in base_ts:
        while k+1 < len(t) and t[k+1] < tau:
            k += 1
        if k+1 >= len(t):
            out.append((float(xy[-1,0]), float(xy[-1,1]), int(orig_idx[-1])))
        else:
            alpha = 0.0 if t[k+1] == t[k] else (tau - t[k]) / (t[k+1] - t[k])
            p = (1 - alpha) * xy[k] + alpha * xy[k+1]
            # interpolate original index for traceability
            idx_val = int(round((1 - alpha) * orig_idx[k] + alpha * orig_idx[k+1]))
            out.append((float(p[0]), float(p[1]), idx_val))

    # Enforce exact endpoints in case of numeric drift
    out[0]  = (float(pts[0][0]),  float(pts[0][1]),  int(pts[0][2]))
    out[-1] = (float(pts[-1][0]), float(pts[-1][1]), int(pts[-1][2]))
    return out[:target]

def resample_keypoints(
    points: List[Tuple[float, float, int]], 
    target_count: int
) -> List[Tuple[float, float, int]]:
    """
    Resample keypoints to target_count while preserving sequential path order.
    Uses arc-length parameterization to maintain natural progression.
    
    Args:
        points: List of (x, y, idx) tuples in original order.
        target_count: Target number of points (e.g., 25).
    
    Returns:
        Resampled list of (x, y, idx) in preserved order.
    """
    if not points:
        return [(0.0, 0.0, 0)] * target_count  # Pad if empty
    
    if len(points) <= 1:
        # If only one point, pad with copies
        if len(points) == 1:
            return [points[0]] * target_count
        else:
            return [(0.0, 0.0, 0)] * target_count
    
    # Extract coordinates
    coords = np.array([(p[0], p[1]) for p in points])
    
    if len(coords) == target_count:
        return [(float(x), float(y), p[2]) for (x, y), p in zip(coords, points)]
    
    # Compute cumulative distances along path for parameterization
    diffs = np.diff(coords, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    cum_dist = np.cumsum(np.insert(distances, 0, 0))  # Cumulative arc length
    total_length = cum_dist[-1]
    
    if total_length == 0:
        # Degenerate case: uniform spacing by index
        indices = np.linspace(0, len(points)-1, target_count)
        resampled = np.zeros((target_count, 2))
        for i, idx in enumerate(indices):
            if idx == 0:
                resampled[i] = coords[0]
            elif idx == len(points)-1:
                resampled[i] = coords[-1]
            else:
                int_idx = int(idx)
                frac = idx - int_idx
                if int_idx + 1 < len(coords):
                    prev, next_ = int_idx, int_idx + 1
                    resampled[i] = coords[prev] * (1 - frac) + coords[next_] * frac
                else:
                    resampled[i] = coords[-1]  # fallback to last point
    else:
        # Interpolate using arc-length parameter
        t_new = np.linspace(0, total_length, target_count)
        # Use linear interpolation for robustness
        interp_x = interp1d(cum_dist, coords[:, 0], kind='linear', bounds_error=False, fill_value='extrapolate')
        interp_y = interp1d(cum_dist, coords[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')
        resampled = np.column_stack([interp_x(t_new), interp_y(t_new)])
    
    # Assign new indices (original idx interpolated for continuity)
    orig_indices = np.array([p[2] for p in points])
    interp_idx = interp1d(np.arange(len(points)), orig_indices, kind='linear', bounds_error=False, fill_value='extrapolate')
    new_indices = interp_idx(np.linspace(0, len(points)-1, target_count)).astype(int)
    
    return [(float(resampled[i, 0]), float(resampled[i, 1]), int(new_indices[i])) for i in range(target_count)]

def resample_keypoints_iterative(points, target=51):
    """
    Reamostra pontos para uma contagem alvo, dividindo iterativamente o segmento mais longo.
    Isso garante que os pontos sejam distribuídos por todo o caminho e evita
    a concentração em uma área. Preserva todos os pontos originais.
    
    Args:
        points: list[(x, y, idx)] - DEVE estar em ordem de caminho (classificado por idx)
        target: Número alvo de pontos (ex: 51)
    
    Returns:
        list[(x, y, idx)] de comprimento == target
    """
    import heapq
    import numpy as np

    current_points = list(points) # Pontos já devem estar em ordem de caminho
    n = len(current_points)
    
    if n >= target:
        if n == target:
            return current_points
        # O downsampling é tratado pelo chamador, mas como segurança:
        indices = np.linspace(0, n - 1, target).astype(int)
        return [current_points[i] for i in indices]

    num_to_add = target - n
    
    # Helper para calcular o comprimento do segmento
    def get_seg_len(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Fila de prioridade para armazenar segmentos, priorizados por comprimento (max-heap)
    # Formato do item: (-comprimento, p1, p2)
    pq = []

    # Inicializa a fila com todos os segmentos originais
    for i in range(n - 1):
        p1 = current_points[i]
        p2 = current_points[i+1]
        length = get_seg_len(p1, p2)
        if length > 0:
            heapq.heappush(pq, (-length, p1, p2))

    # Adiciona os novos pontos criados
    newly_added_points = []
    
    for _ in range(num_to_add):
        if not pq:
            break  # Fila vazia (ex: todos os segmentos têm comprimento 0)

        # 1. Pega o segmento mais longo
        neg_len, p1, p2 = heapq.heappop(pq)
        
        # 2. Cria o novo ponto central
        # Também interpolamos o original_idx para rastreabilidade
        mid_x = (p1[0] + p2[0]) / 2.0
        mid_y = (p1[1] + p2[1]) / 2.0
        mid_idx = int((p1[2] + p2[2]) / 2.0) # Interpolação simples de índice
        new_midpoint = (mid_x, mid_y, mid_idx)
        newly_added_points.append(new_midpoint)

        # 3. Recalcula a importância (comprimento) e adiciona os dois novos segmentos de volta
        len1 = get_seg_len(p1, new_midpoint)
        len2 = get_seg_len(new_midpoint, p2)
        
        if len1 > 0:
            heapq.heappush(pq, (-len1, p1, new_midpoint))
        if len2 > 0:
            heapq.heappush(pq, (-len2, new_midpoint, p2))

    # Combina os pontos originais e os novos pontos
    final_points = current_points + newly_added_points
    
    return final_points

def pad_keypoints(
    keypoints: List[Tuple[float, float]], 
    target_count: int,
    pad_value: Tuple[float, float] = (0.0, 0.0)
) -> List[Tuple[float, float, int]]:
    """
    Pad keypoints to target_count with visibility flags.
    
    Args:
        keypoints: List of (x, y) or (x, y, visibility) tuples 
        target_count: Target number of keypoints
        pad_value: Default value for padding
    
    Returns:
        List of (x, y, visibility) tuples
        visibility: 0 (not labeled), 2 (visible)
    """
    result = []
    for keypoint in keypoints:
        if len(keypoint) == 2:  # (x, y)
            result.append((keypoint[0], keypoint[1], 2))
        elif len(keypoint) == 3:  # (x, y, visibility)
            result.append((keypoint[0], keypoint[1], keypoint[2]))
        else:
            result.append((0.0, 0.0, 0))  # default to invisible
    
    # Pad with invisible keypoints
    while len(result) < target_count:
        result.append((pad_value[0], pad_value[1], 0))  # vis=0: not labeled
    
    return result[:target_count]  # Truncate if too many

# Import merge functionality
try:
    from merge_json import batch_merge_all
except ImportError:
    print("Warning: merge_json.py not found. JSON merging functionality will be disabled.")
    batch_merge_all = None

# Helper functions for deterministic pose construction from plotted points
def order_left_to_right(points):
    """
    Sort points by x ascending; tie-break by y, then original draw index for determinism.
    
    Args:
        points: list of (x, y, original_idx) tuples
    
    Returns:
        sorted list of (x, y, original_idx) tuples
    """
    # points: list[(x, y, idx)]
    # Sort by x ascending; tie-break by y, then original idx for determinism
    pts = sorted(points, key=lambda p: (p[0], p[1], p[2]))
    return pts

def curvature_importance(points):
    """
    Compute importance per vertex from turn angle and arc length.
    
    Args:
        points: list of (x, y, original_idx) tuples
    
    Returns:
        numpy array of importance values for each point
    """
    import numpy as np
    if len(points) < 3:
        return np.ones(len(points))
    xy = np.array([[p[0], p[1]] for p in points], float)
    v = xy[1:] - xy[:-1]
    ln = (np.linalg.norm(v, axis=1) + 1e-8)[:, None]
    v = v / ln
    cos = np.clip((v[:-1] * v[1:]).sum(axis=1), -1.0, 1.0)
    ang = np.arccos(cos)
    imp = np.zeros(len(points))
    # distribute angle to adjacent vertices
    imp[1:-1] += ang
    # add arc length so long bends get weight
    seglen = (ln.flatten())
    imp[:-1] += seglen
    imp[1:]  += seglen
    # ensure endpoints are kept
    imp[0]  += 1e6
    imp[-1] += 1e6
    return imp

def build_51_from_plotted(points):
    """
    Constrói exatamente 51 keypoints a partir de pontos plotados.
    
    MODIFICADO: Usa interpolação iterativa (dividindo o segmento mais longo) 
    para upsampling (n < 51) para evitar a concentração de pontos, preservando 
    todos os pontos originais. Garante que a saída final seja classificada 
    pela coordenada x.
    Usa downsampling baseado em importância para (n > 51).
    
    Args:
        points: list of (x, y, original_idx) tuples (esperado em ordem de caminho)
    
    Returns:
        list of exactly 51 (x, y, idx) tuples
    """
    
    # Precisamos de duas versões da lista de pontos:
    # 1. Classificada por caminho (índice original) para UPSAMPLING
    pts_by_path = sorted(points, key=lambda p: p[2])
    n = len(pts_by_path)
    
    if n == 0:
        return [(0.0, 0.0, 0)] * 51
    if n == 1:
        return [pts_by_path[0]] * 51

    # --- Caso 1: Downsample (n > 51) ---
    # Esta lógica precisa de pontos classificados pela coordenada X
    if n > 51:
        import numpy as np
        # Cria a lista classificada por x APENAS quando necessário
        pts_by_x = order_left_to_right(points) 
        imp = curvature_importance(pts_by_x)
        # Sempre inclui pontos finais, depois escolhe os (51-2) melhores internos por importância
        keep = {0, n-1}
        interior = list(range(1, n-1))
        interior_sorted = sorted(interior, key=lambda i: imp[i], reverse=True)
        for i in interior_sorted[:51 - 2]:
            keep.add(i)
        sel = sorted(list(keep))
        # Retorna os pontos originais selecionados (que são da lista classificada por x)
        return [pts_by_x[i] for i in sel]

    # --- Caso 2: Upsample (n < 51) ---
    # Esta lógica precisa de pontos classificados por CAMINHO (índice original)
    if n < 51:
        import numpy as np
        
        # 1. Salva os pontos originais (usando a lista classificada por caminho)
        original_points_set = set((p[0], p[1]) for p in pts_by_path)
        
        # 2. Chama a nova função de reamostragem iterativa
        # Passa os pontos na ORDEM DE CAMINHO correta (classificados por p[2])
        interpolated_pts = resample_keypoints_iterative(
            pts_by_path, 
            target=51
        )
        
        # 3. Verificação (Opcional, mas mantida)
        if GENERATION_CONFIG.get('debug_coords', False):
            final_points_set = set((p[0], p[1]) for p in interpolated_pts)
            preserved_count = 0
            missing_originals = []
            
            for orig_pt in original_points_set:
                # Use uma tolerância para comparação de floats
                is_present = any(
                    np.isclose(orig_pt[0], final_pt[0]) and np.isclose(orig_pt[1], final_pt[1])
                    for final_pt in final_points_set
                )
                if is_present:
                    preserved_count += 1
                else:
                    missing_originals.append(orig_pt)

            print(f"DEBUG [UPSAMPLE-VERIFY-ITERATIVE] Preserved {preserved_count} of {n} original points.")
            if missing_originals:
                print(f"DEBUG [UPSAMPLE-VERIFY-ITERATIVE] MISSING {len(missing_originals)} ORIGINALS: {missing_originals[:5]}...")
            elif preserved_count == n:
                print("DEBUG [UPSAMPLE-VERIFY-ITERATIVE] All original points successfully preserved.")
        
        # 4. Classificação Final: Classifica por coordenadas x (da esquerda para a direita)
        # Este é o passo final, conforme solicitado, para garantir a ordem correta.
        final_sorted_pts = sorted(interpolated_pts, key=lambda p: (p[0], p[1], p[2]))
        
        return final_sorted_pts

    # --- Caso 3: n == 51 ---
    # Retorna os pontos classificados pela coordenada X
    pts_by_x = order_left_to_right(points)
    return pts_by_x

# ===================================================================================
# ==                               CONFIGURATION                                   ==
# ===================================================================================
GENERATION_CONFIG = {
    "debug_mode": False,
    "debug_annotations": False,  # Detailed annotation processing logs
    "debug_artists": False,      # Detailed artist processing logs
    "debug_coords": False,       # Detailed coordinate transformation logs
    "num_images": 100, 
    "output_dir": "train",
    "seed": 42,

"CLASS_MAP_BAR": {
    "0": "chart",
    "1": "bar",
    "2": "axis_title",
    "3": "significance_marker",
    "4": "error_bar",
    "5": "legend",
    "6": "chart_title",
    "7": "data_label",
    "8": "axis_labels"
  },
  "CLASS_MAP_PIE_OBJ": {
    "0": "chart",
    "1": "wedge",
    "2": "legend",
    "3": "chart_title",
    "4": "data_label",
    "5": "connector_line"
},
"CLASS_MAP_PIE_POSE": {
    "0": "slice_boundary",
},
"CLASS_MAP_LINE_OBJ": {
    "0": "chart",
    "1": "line_segment",
    "2": "axis_title",
    "3": "legend",
    "4": "chart_title",
    "5": "data_label",
    "6": "axis_labels"
},
"CLASS_MAP_LINE_POSE": {
    "0": "line_boundary",
},
  "CLASS_MAP_SCATTER": {
    "0": "chart",
    "1": "data_point",
    "2": "axis_title",
    "3": "significance_marker",
    "4": "error_bar",
    "5": "legend",
    "6": "chart_title",
    "7": "data_label",
    "8": "axis_labels"
  },
  "CLASS_MAP_BOX": {
    "0": "chart",
    "1": "box",
    "2": "axis_title",
    "3": "significance_marker",
    "4": "range_indicator",
    "5": "legend",
    "6": "chart_title",
    "7": "median_line",
    "8": "axis_labels",
    "9": "outlier"
  },
  "CLASS_MAP_HISTOGRAM": {
    "0": "chart",
    "1": "bar",
    "2": "axis_title",
    "3": "legend",
    "4": "chart_title",
    "5": "data_label",
    "6": "axis_labels"
  },
  "CLASS_MAP_HEATMAP": {
    "0": "chart",
    "1": "cell",
    "2": "axis_title",
    "3": "color_bar",
    "4": "legend",
    "5": "chart_title",
    "6": "data_label",
    "7": "axis_labels",
    "8": "significance_marker"
  },
  "CLASS_MAP_AREA_POSE": {
    "0": "area_boundary",
  },
  "CLASS_MAP_AREA_OBJ": {
    "0": "chart",
    "1": "axis_title",
    "2": "legend",
    "3": "chart_title",
    "4": "data_label",
    "5": "axis_labels"
  },

    "scenario_weights": {
        "single": 80,
        "overlay": 0,
        "multi": 20,
    },

    "chart_types": {
        "bar":       {"weight": 13, "enabled": True},
        "line":      {"weight": 13, "enabled": True},
        "scatter":   {"weight": 13, "enabled": True},
        "box":       {"weight": 13,  "enabled": True},
        "pie":       {"weight": 12, "enabled": True},
        "area":      {"weight": 12, "enabled": True},
        "histogram": {"weight": 12, "enabled": True},
        "heatmap":   {"weight": 12,  "enabled": True},
    },
    
    "bar_chart_config": {
        "scientific_ratio": 0.6,
        "styles": {
            "standard":             {"weight": 30},
            "compare_side_by_side": {"weight": 25},
            "stacked":              {"weight": 20},
            "touching":             {"weight": 15},
            "3d_effect":            {"weight": 10},
        },
        "patterns": {
            "none":    {"weight": 50}, "hatch":   {"weight": 20}, "hollow":  {"weight": 10},
            "striped": {"weight": 10}, "dotted":  {"weight": 10},
        }
    },
    
    "realism_effects": {
        "blur":               {"p": 0.1, "params": {"radius_range": (0.25, 0.50)}},
        "motion_blur":        {"p": 0.15, "params": {"radius_range": (2, 5), "angle_range": (0, 360)}},
        "low_res":            {"p": 0.15, "params": {"scale_range": (0.25, 0.4)}},
        "noise":              {"p": 0.05, "params": {"sigma_range": (1, 4)}},
        "jpeg_compression":   {"p": 0.20, "params": {"quality_range": (50, 90)}},
        "pixelation":         {"p": 0.05, "params": {"factor_options": [2, 3]}},
        "posterize":          {"p": 0.05, "params": {"color_options": [16, 32, 64]}},
        "color_variation":    {"p": 0.05, "params": {"shift_range": (0.97, 1.03)}},
        "ui_chrome":          {"p": 0.05, "params": {}},
        "watermark":          {"p": 0.05, "params": {"opacity_range": (0.04, 0.12)}},
        "vignette":           {"p": 0.05, "params": {}},
        "scanner_streaks":    {"p": 0.05, "params": {}},
        "clipping":           {"p": 0.0, "params": {"clip_range_pct": (0.01, 0.04)}},
        "printing_artifacts": {"p": 0.05, "params": {"texture_alpha": (0.05, 0.1), "blur_radius": (0.2, 0.4)}},
        "mouse_cursor":       {"p": 0.05, "params": {}},
        "text_degradation":   {"p": 0.05, "params": {"blur_radius_range": (0.4, 0.6), "pixelate_scale_options": [2, 3]}},
        "grid_occlusion":     {"p": 0.0, "params": {}},
        "scan_rotation":      {"p": 0.0, "params": {"angle_range": (-1, 1)}},
        "grayscale":          {"p": 0.05, "params": {}},
        "perspective":        {"p": 0.0, "params": {"magnitude": 0.5}},
    },
    
    # Post-processing options
    "merge_json_files": True,  # Set to True to merge the 3 JSON files into 1 comprehensive JSON
}

# Chart-type-specific class maps
CHART_CLASS_MAPS = {
    'bar': GENERATION_CONFIG['CLASS_MAP_BAR'],
    'scatter': GENERATION_CONFIG['CLASS_MAP_SCATTER'],
    'box': GENERATION_CONFIG['CLASS_MAP_BOX'],
    'histogram': GENERATION_CONFIG['CLASS_MAP_HISTOGRAM'],
    'heatmap': GENERATION_CONFIG['CLASS_MAP_HEATMAP'],
    'area_obj': GENERATION_CONFIG['CLASS_MAP_AREA_OBJ'], 
    'area_pose': GENERATION_CONFIG['CLASS_MAP_AREA_POSE'],
    'pie_obj': GENERATION_CONFIG['CLASS_MAP_PIE_OBJ'],
    'pie_pose': GENERATION_CONFIG['CLASS_MAP_PIE_POSE'],
    'line_obj': GENERATION_CONFIG['CLASS_MAP_LINE_OBJ'],
    'line_pose': GENERATION_CONFIG['CLASS_MAP_LINE_POSE']   
}

# ===================================================================================
# == UTILITY FUNCTIONS
# ===================================================================================

def is_float(text):
    try:
        float(text)
        return True
    except (ValueError, TypeError):
        return False

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def bbox_to_yolo_norm(x0, y0, x1, y1, img_w, img_h):
    if img_w == 0 or img_h == 0: 
        return 0, 0, 0, 0
    dw = 1. / img_w
    dh = 1. / img_h
    x = (x0 + x1) / 2.0
    y = (y0 + y1) / 2.0
    w = x1 - x0
    h = y1 - y0
    return x * dw, y * dh, w * dw, h * dh

def bbox_to_xyxy(bbox, img_h):
    """Convert matplotlib bbox to [x0, y0, x1, y1] xyxy format"""
    if hasattr(bbox, 'extents'):
        x0, y0, x1, y1 = bbox.extents
    else:
        x0, y0, x1, y1 = bbox
    return [int(x0), int(img_h - y1), int(x1), int(img_h - y0)]

def bbox_to_xyxy_absolute(bbox, img_h):
    """Convert matplotlib bbox to [x0, y0, x1, y1] absolute xyxy format"""
    if hasattr(bbox, 'extents'):
        x0, y0, x1, y1 = bbox.extents
    else:
        x0, y0, x1, y1 = bbox
    # Convert from matplotlib coordinates to image coordinates
    abs_y0 = int(img_h - y1)
    abs_y1 = int(img_h - y0)
    return [int(x0), abs_y0, int(x1), abs_y1]

def create_reverse_class_map(cls_map):
    """Create reverse mapping: class_name -> class_id"""
    return {v: k for k, v in cls_map.items()}

def has_non_background_pixels(label_artist, fig, none, ax, threshold=1):
    """
    Check if label region contains pixels of different colors.
    Stops immediately upon finding color variation.
    
    Args:
        label_artist: matplotlib Text artist (axis label)
        fig: matplotlib figure
        ax: matplotlib axes
        threshold: pixel difference threshold (0-255 scale)
    
    Returns:
        bool: True if color variation found, False if all pixels same color
    """
    import numpy as np
    
    try:
        renderer = fig.canvas.get_renderer()
        if renderer is None:
            return True
        
        bbox = label_artist.get_window_extent(renderer)
        x0, y0, x1, y1 = int(bbox.x0), int(bbox.y0), int(bbox.x1), int(bbox.y1)
        
        if x1 <= x0 or y1 <= y0 or bbox.width < 1 or bbox.height < 1:
            return False
        
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        h, w = int(fig.bbox.height), int(fig.bbox.width)
        buf = buf.reshape((h, w, 4))
        
        x0 = max(0, min(x0, w-1))
        x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h-1))
        y1 = max(0, min(y1, h))
        
        if x1 <= x0 or y1 <= y0:
            return False
        
        roi = buf[h-y1:h-y0, x0:x1, :3]
        pixels = roi.reshape(-1, 3)
        
        if pixels.shape[0] == 0:
            return False
        
        reference_pixel = pixels[0].astype(np.int16)
        step = max(1, len(pixels) // 100)
        
        for i in range(step, len(pixels), step):
            diff = np.abs(pixels[i].astype(np.int16) - reference_pixel)
            if np.any(diff > threshold):
                return True
        
        if len(pixels) > 20:
            critical_indices = [0, 1, 2, len(pixels)//4, len(pixels)//2, 
                              3*len(pixels)//4, -3, -2, -1]
            for i in critical_indices:
                if i < len(pixels):
                    diff = np.abs(pixels[i].astype(np.int16) - reference_pixel)
                    if np.any(diff > threshold):
                        return True
        
        return False
        
    except Exception as e:
        return True



def get_granular_annotations(fig, chart_info_map, cls_map):
    """
    ENHANCED VERSION WITH CRITICAL BUG FIXES
    
    Key fixes:
    1. Proper artist visibility validation
    2. Robust bounding box calculation with error handling
    3. Enhanced artist type detection for Rectangle objects
    4. Improved add_unique_annotation filtering
    5. Comprehensive debug logging
    6. Fixed class map lookup to handle integer keys correctly
    """
    
    # CRITICAL FIX: Create reverse map for string lookups
    reverse_map = create_reverse_class_map(cls_map)
    
    # CRITICAL FIX: Ensure renderer is ready
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if renderer is None:
        print("WARNING: Could not obtain renderer, annotations will be empty")
        return []
    
    annotations = []
    fig_bbox = fig.get_window_extent(renderer)
    seen_annotations = set()
    
    def add_unique_annotation(class_id, bbox, text=None):
        """FIXED: More robust bbox validation"""
        if bbox is None:
            if GENERATION_CONFIG.get('debug_mode', False):
                print(f"DEBUG: Skipping annotation - bbox is None")
            return False
            
        # Handle different bbox types
        try:
            if hasattr(bbox, 'width') and hasattr(bbox, 'height'):
                width, height = bbox.width, bbox.height
                x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
            elif hasattr(bbox, 'extents'):
                x0, y0, x1, y1 = bbox.extents
                width, height = x1 - x0, y1 - y0
            else:
                x0, y0, x1, y1 = bbox
                width, height = x1 - x0, y1 - y0
                
        except (AttributeError, ValueError, TypeError) as e:
            if GENERATION_CONFIG.get('debug_mode', False):
                print(f"DEBUG: Bbox extraction failed: {e}")
            return False
        
        # CRITICAL FIX: More lenient size validation
        if width <= 0.5 or height <= 0.5:  # Was > 1, now > 0.5
            if GENERATION_CONFIG.get('debug_mode', False):
                print(f"DEBUG: Bbox too small - w:{width:.2f}, h:{height:.2f}")
            return False
            
        # CRITICAL FIX: More precise deduplication
        key = (class_id, round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1))  # Was 2 decimals, now 1
        if key not in seen_annotations:
            entry = {'class_id': class_id, 'bbox': bbox}
            if text:
                entry['text'] = text
            annotations.append(entry)
            seen_annotations.add(key)
            if GENERATION_CONFIG.get('debug_mode', False):
                print(f"DEBUG: Added annotation - class:{class_id}, bbox:[{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}]")
            return True
        else:
            if GENERATION_CONFIG.get('debug_mode', False):
                print(f"DEBUG: Duplicate annotation filtered - class:{class_id}")
            return False
    
    for ax_idx, ax in enumerate(fig.axes):
        if not ax.get_visible():
            if GENERATION_CONFIG.get('debug_mode', False):
                print(f"DEBUG: AX[{ax_idx}]: Skipping invisible axis")
            continue
            
        chart_info = chart_info_map.get(ax, {})
        chart_type = chart_info.get('chart_type_str')
        
        if GENERATION_CONFIG.get('debug_mode', False):
            print(f"DEBUG: AX[{ax_idx}]: Processing chart_type={chart_type}")
        
        # CRITICAL FIX: Enhanced Chart Title detection
        if 'chart_title' in reverse_map:
            title = ax.title
            if title and title.get_visible() and title.get_text().strip():
                try:
                    title_bbox = title.get_window_extent(renderer)
                    if add_unique_annotation(reverse_map['chart_title'], title_bbox, text=title.get_text().strip()):
                        if GENERATION_CONFIG.get('debug_mode', False):
                            print(f"DEBUG: AX[{ax_idx}]: Added chart title annotation")
                except Exception as e:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Chart title bbox failed: {e}")
        
        # CRITICAL FIX: Enhanced Legend detection
        if 'legend' in reverse_map:
            legend = ax.get_legend()
            if legend and legend.get_visible():
                try:
                    # Check if the legend has non-background pixels before adding annotation
                    if has_non_background_pixels(legend, fig, ax, ax.get_facecolor(), threshold=5):
                        legend_bbox = legend.get_window_extent(renderer)
                        if add_unique_annotation(reverse_map['legend'], legend_bbox):
                            if GENERATION_CONFIG.get('debug_mode', False):
                                print(f"DEBUG: AX[{ax_idx}]: Added legend annotation")
                    else:
                        if GENERATION_CONFIG.get('debug_mode', False):
                            print(f"DEBUG: AX[{ax_idx}]: Legend empty (no pixels), skipping")
                except Exception as e:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Legend bbox failed: {e}")
        
        # CRITICAL FIX: Enhanced Axis Title detection
        if 'axis_title' in reverse_map:
            # X-axis title
            if ax.xaxis.label.get_visible() and ax.xaxis.label.get_text().strip():
                try:
                    xlabel_bbox = ax.xaxis.label.get_window_extent(renderer)
                    if add_unique_annotation(reverse_map['axis_title'], xlabel_bbox, text=ax.xaxis.label.get_text().strip()):
                        if GENERATION_CONFIG.get('debug_mode', False):
                            print(f"DEBUG: AX[{ax_idx}]: Added x-axis title annotation")
                except Exception as e:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: X-axis title bbox failed: {e}")
            
            # Y-axis title  
            if ax.yaxis.label.get_visible() and ax.yaxis.label.get_text().strip():
                try:
                    ylabel_bbox = ax.yaxis.label.get_window_extent(renderer)
                    if add_unique_annotation(reverse_map['axis_title'], ylabel_bbox, text=ax.yaxis.label.get_text().strip()):
                        if GENERATION_CONFIG.get('debug_mode', False):
                            print(f"DEBUG: AX[{ax_idx}]: Added y-axis title annotation")
                except Exception as e:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Y-axis title bbox failed: {e}")
        
        # CRITICAL FIX: Enhanced Axis Labels detection
        if 'axis_labels' in reverse_map:
            scale_axis_info = chart_info.get('scale_axis_info', {})
            primary_scale_axis = scale_axis_info.get('primary_scale_axis', 'y')
            bg_color = ax.get_facecolor()
            # X-axis labels
            x_labels_added = 0
            for label in ax.get_xticklabels():
                if label.get_visible() and label.get_text().strip():
                    try:
                        label_bbox = label.get_window_extent(renderer)
                        if add_unique_annotation(reverse_map['axis_labels'], label_bbox, text=label.get_text().strip()):
                            x_labels_added += 1
                    except Exception as e:
                        if GENERATION_CONFIG.get('debug_mode', False):
                            print(f"DEBUG: AX[{ax_idx}]: X-label bbox failed: {e}")
                            
            # Y-axis labels
            y_labels_added = 0        
            for label in ax.get_yticklabels():
                if label.get_visible() and label.get_text().strip():
                    try:
                        label_bbox = label.get_window_extent(renderer)
                        if add_unique_annotation(reverse_map['axis_labels'], label_bbox, text=label.get_text().strip()):
                            y_labels_added += 1
                    except Exception as e:
                        if GENERATION_CONFIG.get('debug_mode', False):
                            print(f"DEBUG: AX[{ax_idx}]: Y-label bbox failed: {e}")
                            
            if GENERATION_CONFIG.get('debug_mode', False):
                print(f"DEBUG: AX[{ax_idx}]: Added {x_labels_added} x-labels, {y_labels_added} y-labels")
        
        # CRITICAL FIX: Enhanced Bar chart data artist processing
        if chart_type == 'bar' and 'bar' in reverse_map:
            data_artists = chart_info.get('data_artists', [])
            bars_added = 0
            
            if GENERATION_CONFIG.get('debug_mode', False):
                print(f"DEBUG: AX[{ax_idx}]: Processing {len(data_artists)} bar data artists")
            
            for artist_idx, artist in enumerate(data_artists):
                if artist is None:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Artist {artist_idx} is None")
                    continue
                    
                # Enhanced visibility check
                try:
                    is_visible = artist.get_visible()
                except:
                    is_visible = True  # Assume visible if check fails
                    
                if not is_visible:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Artist {artist_idx} not visible")
                    continue
                
                # CRITICAL FIX: More robust Rectangle detection
                is_rectangle = (isinstance(artist, patches.Rectangle) or 
                              str(type(artist).__name__) == 'Rectangle' or
                              hasattr(artist, 'get_x') and hasattr(artist, 'get_y') and 
                              hasattr(artist, 'get_width') and hasattr(artist, 'get_height'))
                
                if is_rectangle:
                    try:
                        # CRITICAL FIX: Enhanced bbox extraction
                        artist_bbox = artist.get_window_extent(renderer)
                        if artist_bbox and add_unique_annotation(reverse_map['bar'], artist_bbox):
                            bars_added += 1
                            if GENERATION_CONFIG.get('debug_mode', False):
                                print(f"DEBUG: AX[{ax_idx}]: Added bar annotation #{artist_idx}")
                    except Exception as e:
                        if GENERATION_CONFIG.get('debug_mode', False):
                            print(f"DEBUG: AX[{ax_idx}]: Bar bbox failed for artist {artist_idx}: {e}")
                else:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Artist {artist_idx} type: {type(artist).__name__} - not Rectangle")
                        
            if GENERATION_CONFIG.get('debug_mode', False):
                print(f"DEBUG: AX[{ax_idx}]: Successfully added {bars_added} bar annotations")

        elif chart_type == 'histogram':
            # HISTOGRAM - FIX AXIS TITLE vs AXIS LABELS CONFUSION
            debug = GENERATION_CONFIG.get('debug_mode', False)
            if debug:
                print(f"DEBUG [AX{ax_idx}] HISTOGRAM axis processing override")
            
            if "bar" in reverse_map:
                dataartists = chart_info.get("dataartists", [])
                barsadded = 0
                if debug:
                    print(f"DEBUG: AX{ax_idx} Processing {len(dataartists)} histogram bar patches")
                
                for artistidx, artist in enumerate(dataartists):
                    if artist is None:
                        continue
                    try:
                        isvisible = artist.get_visible()
                    except:
                        isvisible = True
                    
                    if not isvisible:
                        continue
                    
                    # Histogram patches are always Rectangle objects
                    if isinstance(artist, patches.Rectangle):
                        try:
                            artistbbox = artist.get_window_extent(renderer)
                            if artistbbox and add_unique_annotation(reverse_map["bar"], artistbbox):
                                barsadded += 1
                                if debug:
                                    print(f"DEBUG: AX{ax_idx} Added histogram bar {artistidx}")
                        except Exception as e:
                            if debug:
                                print(f"DEBUG: AX{ax_idx} Histogram bar bbox failed: {e}")
                
                if debug:
                    print(f"DEBUG: AX{ax_idx} Total histogram bars annotated: {barsadded}")
                    
            # Override the general axis processing for histograms
            if 'axis_title' in reverse_map:
                titles_added = 0
                # Axis TITLES (xlabel/ylabel - these should be axis_title)
                if ax.xaxis.label.get_visible() and ax.xaxis.label.get_text().strip():
                    try:
                        xlabel_bbox = ax.xaxis.label.get_window_extent(renderer)
                        if add_unique_annotation(reverse_map['axis_title'], xlabel_bbox, text=ax.xaxis.label.get_text().strip()):
                            titles_added += 1
                            if debug:
                                print(f"DEBUG [AX{ax_idx}] Added histogram X-axis TITLE")
                    except Exception as e:
                        if debug:
                            print(f"DEBUG [AX{ax_idx}] Histogram X-title error: {e}")
                
                # X-axis title
                if ax.yaxis.label.get_visible() and ax.yaxis.label.get_text().strip():
                    try:
                        ylabel_bbox = ax.yaxis.label.get_window_extent(renderer)
                        if add_unique_annotation(reverse_map['axis_title'], ylabel_bbox, text=ax.yaxis.label.get_text().strip()):
                            titles_added += 1
                            if debug:
                                print(f"DEBUG [AX{ax_idx}] Added histogram Y-axis TITLE")
                    except Exception as e:
                        if debug:
                            print(f"DEBUG [AX{ax_idx}] Histogram Y-title error: {e}")
                
                if debug:
                    print(f"DEBUG [AX{ax_idx}] HISTOGRAM axis titles: {titles_added}")
                # Y-axis title
            
            if 'axis_labels' in reverse_map:
                labels_added = 0
                bg_color = ax.get_facecolor()
                # Axis LABELS (tick labels - these should be axis_labels)
                for label in ax.get_xticklabels():
                    if label.get_visible() and label.get_text().strip():
                        try:
                            label_bbox = label.get_window_extent(renderer)
                            if add_unique_annotation(reverse_map['axis_labels'], label_bbox, text=label.get_text().strip()):
                                labels_added += 1
                        except Exception as e:
                            if debug:
                                print(f"DEBUG [AX{ax_idx}] Histogram X-label error: {e}")
                
                # X-axis tick labels
                for label in ax.get_yticklabels():
                    if label.get_visible() and label.get_text().strip():
                        try:
                            label_bbox = label.get_window_extent(renderer)
                            if add_unique_annotation(reverse_map['axis_labels'], label_bbox, text=label.get_text().strip()):
                                labels_added += 1
                        except Exception as e:
                            if debug:
                                print(f"DEBUG [AX{ax_idx}] Histogram Y-label error: {e}")
                
                if debug:
                    print(f"DEBUG [AX{ax_idx}] HISTOGRAM axis labels: {labels_added}")
                # Y-axis tick labels
            
            if 'data_label' in reverse_map:
                data_labels_added = 0
                other_artists = chart_info.get('other_artists', [])
                
                if debug:
                    print(f"DEBUG [AX{ax_idx}] Processing {len(other_artists)} other_artists for data labels")
                
                # Data labels are stored in other_artists (text annotations)
                for artist_idx, artist in enumerate(other_artists):
                    # Check if artist is a Text object
                    if hasattr(artist, 'get_text') and hasattr(artist, 'get_window_extent'):
                        try:
                            # Verify it's visible and has content
                            if artist.get_visible() and artist.get_text().strip():
                                label_bbox = artist.get_window_extent(renderer)
                                
                                if add_unique_annotation(reverse_map['data_label'], label_bbox, text=artist.get_text().strip()):
                                    data_labels_added += 1
                                    if debug:
                                        print(f"DEBUG [AX{ax_idx}] Added data label: '{artist.get_text()}'")
                        except Exception as e:
                            if debug:
                                print(f"DEBUG [AX{ax_idx}] Data label bbox failed for artist {artist_idx}: {e}")
                
                if debug:
                    print(f"DEBUG [AX{ax_idx}] HISTOGRAM data labels: {data_labels_added}")

        # Enhanced Box plot processing with fallback
        elif chart_type == 'box':
            boxplot_dict = chart_info.get('boxplot_dict')
            
            # CRITICAL FIX: Unwrap nested structure from chart.py
            if boxplot_dict and 'boxplot_raw' in boxplot_dict:
                bp_artists = boxplot_dict['boxplot_raw']
            else:
                bp_artists = boxplot_dict
            
            if GENERATION_CONFIG.get('debug_mode', False):
                print(f"DEBUG: AX[{ax_idx}]: Boxplot dict: {bp_artists is not None} | Keys: {list(bp_artists.keys()) if bp_artists else 'None'}")
                print(f"DEBUG: AX[{ax_idx}]: Boxes: {len(bp_artists.get('boxes', [])) if bp_artists else 0}")
                    
            boxes_processed = False
            if bp_artists and bp_artists.get('boxes'):
                if GENERATION_CONFIG.get('debug_mode', False):
                    print(f"DEBUG: AX[{ax_idx}]: Processing boxplot with {len(bp_artists['boxes'])} boxes")
                
                # Boxes
                if 'box' in reverse_map:
                    added = 0
                    for box_artist in bp_artists['boxes']:
                        if box_artist and box_artist.get_visible():
                            try:
                                bbox = box_artist.get_window_extent(renderer)
                                if bbox.width > 0.5 and bbox.height > 0.5 and add_unique_annotation(reverse_map['box'], bbox):
                                    added += 1
                            except Exception as e:
                                if GENERATION_CONFIG.get('debug_mode', False):
                                    print(f"DEBUG: AX[{ax_idx}]: Box bbox error: {e}")
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Added {added} box annotations")
                
                # Medians
                if 'median_line' in reverse_map:
                    added = 0
                    for median in bp_artists.get('medians', []):
                        if median and median.get_visible():
                            try:
                                orig_bbox = median.get_window_extent(renderer)
                                # CRITICAL FIX: Always pad FIRST, then check size
                                pad = 3
                                padded = transforms.Bbox.from_extents(
                                    orig_bbox.x0, orig_bbox.y0 - pad,
                                    orig_bbox.x1, orig_bbox.y1 + pad
                                )
                                
                                # Check padded bbox (not original)
                                if padded.width > 0.5 and padded.height > 0.5:
                                    if add_unique_annotation(reverse_map['median_line'], padded):
                                        added += 1
                            except Exception as e:
                                if GENERATION_CONFIG.get('debug_mode', False):
                                    print(f"DEBUG: AX[{ax_idx}]: Median error: {e}")
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Added {added} median annotations")
                
                # Range indicators (Whiskers and Caps)
                if 'range_indicator' in reverse_map:
                    added = 0
                    num_boxes = len(bp_artists.get('boxes', []))
                    whiskers = bp_artists.get('whiskers', [])
                    caps = bp_artists.get('caps', [])
                    for i in range(num_boxes):
                        try:
                            artists = []
                            idxs = [2*i, 2*i + 1]
                            for idx in idxs:
                                if len(whiskers) > idx and whiskers[idx]:
                                    artists.append(whiskers[idx])
                                if len(caps) > idx and caps[idx]:
                                    artists.append(caps[idx])
                            
                            bboxes = []
                            for art in artists:
                                if art and art.get_visible():
                                    try:
                                        bbox = art.get_window_extent(renderer)
                                        if bbox:
                                            # Always add the bbox; zero-width or zero-height elements should be padded
                                            width, height = bbox.width, bbox.height
                                            if width <= 0.5 or height <= 0.5:
                                                # Pad thin elements to make them detectable
                                                min_size = 3
                                                padded_bbox = transforms.Bbox.from_extents(
                                                    bbox.x0 - min_size/2, bbox.y0 - min_size/2,
                                                    bbox.x1 + min_size/2, bbox.y1 + min_size/2
                                                )
                                                bboxes.append(padded_bbox)
                                            else:
                                                # Normal elements (both width and height > 0.5)
                                                bboxes.append(bbox)
                                    except Exception as e:
                                        if GENERATION_CONFIG.get('debug_mode', False):
                                            print(f"DEBUG: AX[{ax_idx}]: Error processing range indicator artist: {e}")
                                        pass
                            
                            if bboxes:
                                union_bbox = transforms.Bbox.union(bboxes)
                                if union_bbox.width > 0.5 and add_unique_annotation(reverse_map['range_indicator'], union_bbox):
                                    added += 1
                        except Exception as e:
                            if GENERATION_CONFIG.get('debug_mode', False):
                                print(f"DEBUG: AX[{ax_idx}]: Range {i} error: {e}")
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Added {added} range annotations")
                
                # Outliers (Fliers)
                if 'outlier' in reverse_map:
                    added = 0
                    for flier in bp_artists.get('fliers', []):
                        if flier and flier.get_visible():
                            try:
                                xdata, ydata = flier.get_xdata(), flier.get_ydata()
                                for x, y in zip(xdata, ydata):
                                    px, py = ax.transData.transform_point((x, y))
                                    size = 3
                                    bbox = transforms.Bbox.from_extents(
                                        px - size, py - size, px + size, py + size
                                    )
                                    if bbox.width > 0.5 and add_unique_annotation(reverse_map['outlier'], bbox):
                                        added += 1
                            except Exception as e:
                                if GENERATION_CONFIG.get('debug_mode', False):
                                    print(f"DEBUG: AX[{ax_idx}]: Outlier error: {e}")
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Added {added} outlier annotations")
                
                boxes_processed = True
            
            # FALLBACK: If boxplot_dict failed, use data_artists
            if not boxes_processed and 'box' in reverse_map:
                added = 0
                for artist in chart_info.get('data_artists', []):
                    if isinstance(artist, patches.Rectangle) and artist.get_visible():
                        try:
                            bbox = artist.get_window_extent(renderer)
                            if bbox.width > 0.5 and add_unique_annotation(reverse_map['box'], bbox):
                                added += 1
                        except Exception as e:
                            if GENERATION_CONFIG.get('debug_mode', False):
                                print(f"DEBUG: AX[{ax_idx}]: Fallback box error: {e}")
                if GENERATION_CONFIG.get('debug_mode', False):
                    print(f"DEBUG: AX[{ax_idx}]: Fallback added {added} box annotations from data_artists")
        
        # === SCATTER CHARTS - INDIVIDUAL POINT ANNOTATION (ENHANCED DEBUG) ===
        elif chart_type == 'scatter' and 'data_point' in reverse_map:
            data_artists = chart_info.get('data_artists', [])
            points_added = 0
            debug = GENERATION_CONFIG.get('debug_mode', False)

            if debug:
                print(f"DEBUG: AX[{ax_idx}]: SCATTER processing {len(data_artists)} artists")

            for artist_idx, artist in enumerate(data_artists):
                if debug:
                    print(f"DEBUG: AX[{ax_idx}]: Artist {artist_idx}: {type(artist).__name__}")

                if not isinstance(artist, PathCollection):
                    if debug:
                        print(f"DEBUG: AX[{ax_idx}]: Not PathCollection, skipping")
                    continue

                try:
                    offsets = artist.get_offsets()
                    sizes = artist.get_sizes()

                    if debug:
                        print(f"DEBUG: AX[{ax_idx}]: Offsets: {offsets.shape}, Sizes: {sizes}")

                    if len(offsets) == 0:
                        if debug:
                            print(f"DEBUG: AX[{ax_idx}]: No offsets")
                        continue

                    is_uniform_size = (sizes.size == 1)
                    # CRITICAL: Use figure DPI for proper pixel conversion
                    points_to_pixels = 72.0 / fig.dpi

                    if debug:
                        print(f"DEBUG: AX[{ax_idx}]: DPI={fig.dpi}, conversion={points_to_pixels}")
                        print(f"DEBUG: AX[{ax_idx}]: Will process {len(offsets)} points")

                    # CRITICAL: Annotate each point individually
                    for i, (x_data, y_data) in enumerate(offsets):
                        px, py = ax.transData.transform_point((x_data, y_data))
                        s = sizes[0] if is_uniform_size else sizes[i]
                        radius = (np.sqrt(s) / np.sqrt(np.pi)) * points_to_pixels

                        bbox = transforms.Bbox.from_extents(
                            px - radius, py - radius, px + radius, py + radius
                        )

                        if debug and i < 3:
                            print(f"DEBUG: Point {i}: data=({x_data:.2f},{y_data:.2f}) → "
                                  f"display=({px:.1f},{py:.1f}), size={s:.1f}, radius={radius:.2f}")

                        if bbox.width > 0.5 and bbox.height > 0.5:
                            if add_unique_annotation(reverse_map['data_point'], bbox):
                                points_added += 1
                                if debug and i < 3:
                                    print(f"DEBUG: Point {i}: ADDED")
                        else:
                            if debug and i < 3:
                                print(f"DEBUG: Point {i}: TOO SMALL")
                except Exception as e:
                    if debug:
                        print(f"DEBUG: AX[{ax_idx}]: Scatter error: {e}")
                        import traceback
                        traceback.print_exc()

            if debug:
                print(f"DEBUG: AX[{ax_idx}]: SCATTER TOTAL: {points_added} points added")

        # === HEATMAP CHARTS - CELLS AND COLORBAR ===
        elif chart_type == 'heatmap':
            cells_added = 0
            colorbar_added = 0
            debug = GENERATION_CONFIG.get('debug_mode', False)
            
            # FIX #1: AXIS TITLES (MISSING)
            if 'axis_title' in reverse_map:
                titles_added = 0
                if ax.xaxis.label.get_visible() and ax.xaxis.label.get_text().strip():
                    try:
                        xlabel_bbox = ax.xaxis.label.get_window_extent(renderer)
                        if add_unique_annotation(reverse_map['axis_title'], xlabel_bbox, text=ax.xaxis.label.get_text().strip()):
                            titles_added += 1
                    except Exception as e:
                        if debug:
                            print(f"DEBUG: AX[{ax_idx}]: Heatmap X-title error: {e}")
                
                if ax.yaxis.label.get_visible() and ax.yaxis.label.get_text().strip():
                    try:
                        ylabel_bbox = ax.yaxis.label.get_window_extent(renderer)
                        if add_unique_annotation(reverse_map['axis_title'], ylabel_bbox, text=ax.yaxis.label.get_text().strip()):
                            titles_added += 1
                    except Exception as e:
                        if debug:
                            print(f"DEBUG: AX[{ax_idx}]: Heatmap Y-title error: {e}")
                
                if debug:
                    print(f"DEBUG: AX[{ax_idx}]: HEATMAP axis titles: {titles_added}")
            
            # FIX #2: AXIS LABELS (MISSING)
            if 'axis_labels' in reverse_map:
                labels_added = 0
                bg_color = ax.get_facecolor()

                # X-axis tick labels
                for label in ax.get_xticklabels():
                    if label.get_visible() and label.get_text().strip():
                        try:
                            label_bbox = label.get_window_extent(renderer)
                            if add_unique_annotation(reverse_map['axis_labels'], label_bbox, text=label.get_text().strip()):
                                labels_added += 1
                        except Exception as e:
                            if debug:
                                print(f"DEBUG: AX[{ax_idx}]: Heatmap X-label error: {e}")
                
                # Y-axis tick labels
                for label in ax.get_yticklabels():
                    if label.get_visible() and label.get_text().strip():
                        try:
                            label_bbox = label.get_window_extent(renderer)
                            if add_unique_annotation(reverse_map['axis_labels'], label_bbox, text=label.get_text().strip()):
                                labels_added += 1
                        except Exception as e:
                            if debug:
                                print(f"DEBUG: AX[{ax_idx}]: Heatmap Y-label error: {e}")
                
                if debug:
                    print(f"DEBUG: AX[{ax_idx}]: HEATMAP axis labels: {labels_added}")
            
            # FIX #3: CELLS (WRONG API - CORRECTED)
            if 'cell' in reverse_map:
                data_artists = chart_info.get('data_artists', [])
                for artist_idx, artist in enumerate(data_artists):
                    if debug:
                        print(f"DEBUG: AX[{ax_idx}]: Heatmap artist {artist_idx}: {type(artist).__name__}")
                    
                    if isinstance(artist, QuadMesh):
                        try:
                            # CORRECT METHOD: Use get_coordinates() for mesh data
                            coords = artist.get_coordinates()
                            
                            if coords is not None and coords.ndim == 3:
                                rows, cols = coords.shape[0] - 1, coords.shape[1] - 1
                                
                                if debug:
                                    print(f"DEBUG: AX[{ax_idx}]: QuadMesh grid: {rows}x{cols} cells")
                                
                                # Iterate over each cell in the mesh
                                for i in range(rows):
                                    for j in range(cols):
                                        # Get 4 corners of cell (i,j)
                                        cell_corners = np.array([
                                            coords[i, j],      # bottom-left
                                            coords[i+1, j],    # bottom-right
                                            coords[i+1, j+1],  # top-right
                                            coords[i, j+1]     # top-left
                                        ])
                                        
                                        # Transform to display coordinates
                                        display_coords = ax.transData.transform(cell_corners)
                                        
                                        x0, y0 = display_coords.min(axis=0)
                                        x1, y1 = display_coords.max(axis=0)
                                        
                                        bbox = transforms.Bbox.from_extents(x0, y0, x1, y1)
                                        
                                        if bbox.width >= 1.0 and bbox.height >= 1.0:
                                            if add_unique_annotation(reverse_map['cell'], bbox):
                                                cells_added += 1
                                        elif debug and i < 3 and j < 3:
                                            print(f"DEBUG: Cell ({i},{j}) TOO SMALL: {bbox.width:.1f}x{bbox.height:.1f}")
                            
                        except AttributeError:
                            # Fallback for imshow() which creates AxesImage, not QuadMesh
                            try:
                                extent = artist.get_extent()  # (x0, x1, y0, y1)
                                
                                # Get data array dimensions
                                data_array = artist.get_array()
                                if data_array is not None:
                                    if data_array.ndim == 2:
                                        rows, cols = data_array.shape
                                    else:
                                        rows, cols = data_array.shape[0], data_array.shape[1]
                                    
                                    if debug:
                                        print(f"DEBUG: AX[{ax_idx}]: AxesImage grid: {rows}x{cols} cells")
                                    
                                    x0_data, x1_data, y0_data, y1_data = extent
                                    cell_width = (x1_data - x0_data) / cols
                                    cell_height = (y1_data - y0_data) / rows
                                    
                                    for i in range(rows):
                                        for j in range(cols):
                                            # Calculate cell bounds in data coordinates
                                            cell_x0 = x0_data + j * cell_width
                                            cell_x1 = cell_x0 + cell_width
                                            cell_y0 = y0_data + i * cell_height
                                            cell_y1 = cell_y0 + cell_height
                                            
                                            # Transform to display coordinates
                                            pt0 = ax.transData.transform_point((cell_x0, cell_y0))
                                            pt1 = ax.transData.transform_point((cell_x1, cell_y1))
                                            
                                            bbox = transforms.Bbox.from_extents(
                                                pt0[0], pt0[1], pt1[0], pt1[1]
                                            )
                                            
                                            if bbox.width >= 1.0 and bbox.height >= 1.0:
                                                if add_unique_annotation(reverse_map['cell'], bbox):
                                                    cells_added += 1
                            except Exception as e:
                                if debug:
                                    print(f"DEBUG: AX[{ax_idx}]: Heatmap cell fallback error: {e}")
                        
                        except Exception as e:
                            if debug:
                                print(f"DEBUG: AX[{ax_idx}]: Heatmap cell error: {e}")
                    
                    # Handle AxesImage (from imshow)
                    elif hasattr(artist, 'get_extent') and hasattr(artist, 'get_array'):
                        try:
                            extent = artist.get_extent()
                            data_array = artist.get_array()
                            
                            if data_array is not None:
                                if data_array.ndim == 2:
                                    rows, cols = data_array.shape
                                else:
                                    rows, cols = data_array.shape[0], data_array.shape[1]
                                
                                if debug:
                                    print(f"DEBUG: AX[{ax_idx}]: AxesImage: {rows}x{cols} cells")
                                
                                x0_data, x1_data, y0_data, y1_data = extent
                                cell_width = (x1_data - x0_data) / cols
                                cell_height = (y1_data - y0_data) / rows
                                
                                for i in range(rows):
                                    for j in range(cols):
                                        cell_x0 = x0_data + j * cell_width
                                        cell_x1 = cell_x0 + cell_width
                                        cell_y0 = y0_data + i * cell_height
                                        cell_y1 = cell_y0 + cell_height
                                        
                                        pt0 = ax.transData.transform_point((cell_x0, cell_y0))
                                        pt1 = ax.transData.transform_point((cell_x1, cell_y1))
                                        
                                        bbox = transforms.Bbox.from_extents(pt0[0], pt0[1], pt1[0], pt1[1])
                                        
                                        if bbox.width >= 1.0 and bbox.height >= 1.0:
                                            if add_unique_annotation(reverse_map['cell'], bbox):
                                                cells_added += 1
                        
                        except Exception as e:
                            if debug:
                                print(f"DEBUG: AX[{ax_idx}]: AxesImage error: {e}")
                
                if debug:
                    print(f"DEBUG: AX[{ax_idx}]: HEATMAP cells: {cells_added}")
            
            # FIX #4: DATA LABELS (MISSING)
            if 'data_label' in reverse_map:
                data_labels_added = 0
                other_artists = chart_info.get('other_artists', [])
                
                for artist in other_artists:
                    if isinstance(artist, matplotlib.text.Text):
                        if artist.get_visible() and artist.get_text().strip():
                            try:
                                label_bbox = artist.get_window_extent(renderer)
                                if add_unique_annotation(reverse_map['data_label'], label_bbox, text=artist.get_text().strip()):
                                    data_labels_added += 1
                            except Exception as e:
                                if debug:
                                    print(f"DEBUG: AX[{ax_idx}]: Data label error: {e}")
                
                if debug:
                    print(f"DEBUG: AX[{ax_idx}]: HEATMAP data labels: {data_labels_added}")
            
            # FIX #5: COLORBAR (EXISTING CODE - KEEP)
            if 'color_bar' in reverse_map:
                for ax_candidate in fig.axes:
                    if ax_candidate == ax:
                        continue
                    
                    try:
                        ax_bbox = ax_candidate.get_window_extent(renderer)
                        if ax_bbox.height <= 0 or ax_bbox.width <= 0:
                            continue
                        
                        aspect_ratio = ax_bbox.width / ax_bbox.height
                        
                        is_vertical_colorbar = aspect_ratio < 0.3 and ax_bbox.height > 50
                        is_horizontal_colorbar = aspect_ratio > 3.0 and ax_bbox.width > 50
                        
                        if is_vertical_colorbar or is_horizontal_colorbar:
                            if add_unique_annotation(reverse_map['color_bar'], ax_bbox):
                                colorbar_added += 1
                                if debug:
                                    print(f"DEBUG: AX[{ax_idx}]: COLORBAR ADDED ({'vertical' if is_vertical_colorbar else 'horizontal'})")
                                break
                    except Exception as e:
                        if debug:
                            print(f"DEBUG: AX[{ax_idx}]: Colorbar detection error: {e}")
                
                if debug:
                    print(f"DEBUG: AX[{ax_idx}]: HEATMAP colorbar: {colorbar_added}")
        
        # === HISTOGRAM - FIX AXIS TITLE vs AXIS LABELS CONFUSION ===
        elif chart_type == 'histogram':
            # Override the general axis processing for histograms
            debug = GENERATION_CONFIG.get('debug_mode', False)

            if debug:
                print(f"DEBUG: AX[{ax_idx}]: HISTOGRAM axis processing override")

            # CRITICAL: Distinguish between axis TITLES and axis LABELS (tick labels)

            # Axis TITLES (xlabel/ylabel) - these should be 'axis_title'
            if 'axis_title' in reverse_map:
                titles_added = 0

                # X-axis title
                if ax.xaxis.label.get_visible() and ax.xaxis.label.get_text().strip():
                    try:
                        xlabel_bbox = ax.xaxis.label.get_window_extent(renderer)
                        if add_unique_annotation(reverse_map['axis_title'], xlabel_bbox, text=ax.xaxis.label.get_text().strip()):
                            titles_added += 1
                            if debug:
                                print(f"DEBUG: AX[{ax_idx}]: Added histogram X-axis TITLE")
                    except Exception as e:
                        if debug:
                            print(f"DEBUG: AX[{ax_idx}]: Histogram X-title error: {e}")

                # Y-axis title
                if ax.yaxis.label.get_visible() and ax.yaxis.label.get_text().strip():
                    try:
                        ylabel_bbox = ax.yaxis.label.get_window_extent(renderer)
                        if add_unique_annotation(reverse_map['axis_title'], ylabel_bbox, text=ax.yaxis.label.get_text().strip()):
                            titles_added += 1
                            if debug:
                                print(f"DEBUG: AX[{ax_idx}]: Added histogram Y-axis TITLE")
                    except Exception as e:
                        if debug:
                            print(f"DEBUG: AX[{ax_idx}]: Histogram Y-title error: {e}")

                if debug:
                    print(f"DEBUG: AX[{ax_idx}]: HISTOGRAM axis titles: {titles_added}")

            # Axis LABELS (tick labels) - these should be 'axis_labels' 
            if 'axis_labels' in reverse_map:
                labels_added = 0

                # X-axis tick labels
                for label in ax.get_xticklabels():
                    if label.get_visible() and label.get_text().strip():
                        try:
                            label_bbox = label.get_window_extent(renderer)
                            if add_unique_annotation(reverse_map['axis_labels'], label_bbox, text=label.get_text().strip()):
                                labels_added += 1
                        except Exception as e:
                            if debug:
                                print(f"DEBUG: AX[{ax_idx}]: Histogram X-label error: {e}")

                # Y-axis tick labels  
                for label in ax.get_yticklabels():
                    if label.get_visible() and label.get_text().strip():
                        try:
                            label_bbox = label.get_window_extent(renderer)
                            if add_unique_annotation(reverse_map['axis_labels'], label_bbox, text=label.get_text().strip()):
                                labels_added += 1
                        except Exception as e:
                            if debug:
                                print(f"DEBUG: AX[{ax_idx}]: Histogram Y-label error: {e}")

                if debug:
                    print(f"DEBUG: AX[{ax_idx}]: HISTOGRAM axis labels: {labels_added}")

        # Line chart keypoints
        elif chart_type == 'line' and chart_info.get('keypoint_info'):
            for series_kpts in chart_info['keypoint_info']:
                series_idx = series_kpts['series_idx']
                
                # Start/end points
                for pt_type, pt_data in [('line_start', series_kpts['start']), ('line_end', series_kpts['end'])]:
                    if pt_type in reverse_map:
                        x, y, idx = pt_data
                        px, py = ax.transData.transform_point((x, y))
                        bbox = transforms.Bbox.from_extents(px-4, py-4, px+4, py+4)
                        add_unique_annotation(reverse_map[pt_type], bbox)
                
                # Inflection points
                if 'inflection_point' in reverse_map:
                    for x, y, idx in series_kpts['inflections']:
                        px, py = ax.transData.transform_point((x, y))
                        bbox = transforms.Bbox.from_extents(px-3, py-3, px+3, py+3)
                        add_unique_annotation(reverse_map['inflection_point'], bbox)

        # Area chart keypoints (similar structure to line)
        elif chart_type == 'area' and chart_info.get('keypoint_info'):
            for series_kpts in chart_info['keypoint_info']:
                if 'area_start' in reverse_map:
                    x, y, idx = series_kpts['start']
                    px, py = ax.transData.transform_point((x, y))
                    bbox = transforms.Bbox.from_extents(px-4, py-4, px+4, py+4)
                    add_unique_annotation(reverse_map['area_start'], bbox)
                
                if 'inflection_point' in reverse_map:
                    for x, y, idx in series_kpts['inflections']:
                        px, py = ax.transData.transform_point((x, y))
                        bbox = transforms.Bbox.from_extents(px-3, py-3, px+3, py+3)
                        add_unique_annotation(reverse_map['inflection_point'], bbox)

        # Pie chart geometric keypoints
        elif chart_type == 'pie' and chart_info.get('pie_geometry'):
            pie_geo = chart_info['pie_geometry']
            
            # Center point
            if 'center_point' in reverse_map and 'center_point' in pie_geo:
                cx, cy = pie_geo['center_point']
                px, py = ax.transData.transform_point((cx, cy))
                bbox = transforms.Bbox.from_extents(px-5, py-5, px+5, py+5)
                add_unique_annotation(reverse_map['center_point'], bbox)
            
            # Arc boundaries for each wedge
            if 'arc_boundary' in reverse_map:
                for wedge_geo in pie_geo['wedges']:
                    for arc_pt in ['arc_start', 'arc_end', 'arc_mid']:
                        ax_pt, ay_pt = wedge_geo[arc_pt]
                        px, py = ax.transData.transform_point((ax_pt, ay_pt))
                        bbox = transforms.Bbox.from_extents(px-3, py-3, px+3, py+3)
                        add_unique_annotation(reverse_map['arc_boundary'], bbox)
            
            # Wedge centers
            if 'wedge_center' in reverse_map:
                for wedge_geo in pie_geo['wedges']:
                    wx, wy = wedge_geo['wedge_label_point']
                    px, py = ax.transData.transform_point((wx, wy))
                    bbox = transforms.Bbox.from_extents(px-4, py-4, px+4, py+4)
                    add_unique_annotation(reverse_map['wedge_center'], bbox)
        
        # Process other artists for error bars, text annotations, etc.
        other_artists = chart_info.get('other_artists', [])
        if GENERATION_CONFIG.get('debug_mode', False):
            print(f"DEBUG: AX[{ax_idx}]: Processing {len(other_artists)} other artists")
            
        for artist_idx, artist in enumerate(other_artists):
            if artist is None:
                continue
                
            try:
                is_visible = artist.get_visible()
            except:
                is_visible = True
                
            if not is_visible:
                continue
            
            # Error bars
            if hasattr(artist, 'lines') and len(getattr(artist, 'lines', [])) >= 3 and 'error_bar' in reverse_map:
                try:
                    # ErrorbarContainer processing
                    plotline, caplines, barlinecols = artist.lines
                    if barlinecols and caplines:
                        artist_bbox = artist.get_window_extent(renderer)
                        add_unique_annotation(reverse_map['error_bar'], artist_bbox)
                except Exception as e:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Error bar processing failed: {e}")
            
            # Text annotations
            elif hasattr(artist, 'get_text'):
                try:
                    text_content = artist.get_text().strip()
                    if text_content:
                        # Significance markers
                        if text_content in ['*', '**', '***', 'ns', 'a', 'b', 'c', 'd'] and 'significance_marker' in reverse_map:
                            artist_bbox = artist.get_window_extent(renderer)
                            add_unique_annotation(reverse_map['significance_marker'], artist_bbox, text=text_content)
                        # Data labels  
                        elif 'data_label' in reverse_map:
                            # Check if numeric
                            try:
                                float(text_content.replace('%', '').replace(',', ''))
                                artist_bbox = artist.get_window_extent(renderer)
                                add_unique_annotation(reverse_map['data_label'], artist_bbox, text=text_content)
                            except ValueError:
                                pass  # Not numeric
                except Exception as e:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: AX[{ax_idx}]: Text processing failed: {e}")
    
        if GENERATION_CONFIG.get('debug_mode', False) or GENERATION_CONFIG.get('debug_annotations', False):
            print(f"DEBUG: Total annotations generated: {len(annotations)}")
            class_counts = {}
            for ann in annotations:
                class_id = ann['class_id']
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
            print(f"DEBUG: Class distribution: {class_counts}")
        
        return annotations

def filter_overlapping_annotations(annotations, iou_threshold=0.7):
    """Remove annotations with high IoU overlap within the same class."""
    def bbox_iou(bbox1, bbox2):
        x1 = max(bbox1.x0, bbox2.x0)
        y1 = max(bbox1.y0, bbox2.y0)
        x2 = min(bbox1.x1, bbox2.x1)
        y2 = min(bbox1.y1, bbox2.y1)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        inter_area = (x2 - x1) * (y2 - y1)
        bbox1_area = (bbox1.x1 - bbox1.x0) * (bbox1.y1 - bbox1.y0)
        bbox2_area = (bbox2.x1 - bbox2.x0) * (bbox2.y1 - bbox2.y0)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    by_class = defaultdict(list)
    for ann in annotations:
        by_class[ann['class_id']].append(ann)
    
    filtered = []
    for class_id, class_anns in by_class.items():
        class_anns.sort(key=lambda a: (a['bbox'].x1 - a['bbox'].x0) * (a['bbox'].y1 - a['bbox'].y0), reverse=True)
        
        keep = []
        for ann in class_anns:
            is_duplicate = False
            for kept_ann in keep:
                if bbox_iou(ann['bbox'], kept_ann['bbox']) > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(ann)
        
        filtered.extend(keep)
    
    return filtered


def extract_area_pose_annotations_fixed(
    fig, 
    chart_info_map, 
    cls_map_pose: Dict[str, int], 
    img_w: int, 
    img_h: int
) -> List[Dict]:
    """
    Extract YOLO pose annotations for area charts.
    Single class (0): area_boundary with 51 keypoints following top boundary
    CRITICAL FIX: Use only plotted points for pose construction to preserve sharp features.
    """
    keypoint_annotations = []
    
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        
        chart_info = chart_info_map.get(ax, {})
        chart_type = chart_info.get('chart_type_str', '')
        
        if chart_type != 'area':
            continue
        
        keypoint_info = chart_info.get('keypoint_info', [])
        if not keypoint_info:
            continue
        
        for series_kpts in keypoint_info:
            # Use only plotted points for the top boundary of the area fill
            plotted = series_kpts.get('plotted_points', []) or series_kpts.get('fill_top', [])
            if not plotted:
                continue
            
            # Build 51-point pose using only actual plotted vertices (top boundary)
            kpts = build_51_from_plotted(plotted)
            
            # Transform to pixel coordinates
            px_pts = []
            for x, y, _ in kpts:
                px, py = ax.transData.transform_point((x, y))
                px_pts.append((px, img_h - py))  # Y-flip
            
            # Calculate bounding box from the 51 pixel points
            xs, ys = zip(*px_pts)
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            
            # Normalize bbox
            cx = max(0.0, min(1.0, (x0 + x1) / 2 / img_w))
            cy = max(0.0, min(1.0, (y0 + y1) / 2 / img_h))
            w = max(0.0, min(1.0, (x1 - x0) / img_w))
            h = max(0.0, min(1.0, (y1 - y0) / img_h))
            
            # Normalize keypoints
            kp_norm = [
                [
                    max(0.0, min(1.0, x / img_w)),
                    max(0.0, min(1.0, y / img_h)),
                    2  # All visible
                ]
                for x, y in px_pts
            ]
            
            if GENERATION_CONFIG.get('debug_coords', False):
                print(f"DEBUG [AREA-POSE] Series: Built 51 from {len(plotted)} plotted points (top boundary)")
                if len(plotted) > 0:
                    print(f"DEBUG [AREA-POSE] Series: Original x range: [{plotted[0][0]:.2f}, {plotted[-1][0]:.2f}]")
                if len(kp_norm) > 0:
                    x_norms = [kp[0] for kp in kp_norm]
                    print(f"DEBUG [AREA-POSE] Series: Final x range: [{min(x_norms):.4f}, {max(x_norms):.4f}]")
                    # Check monotonicity
                    is_monotonic = all(x_norms[i] <= x_norms[i+1] for i in range(len(x_norms)-1))
                    print(f"DEBUG [AREA-POSE] Series: X coordinates monotonic: {is_monotonic}")
            
            keypoint_annotations.append({
                'class_id': 0,  # area_boundary
                'bbox': (cx, cy, w, h),
                'keypoints': kp_norm
            })
    
    return keypoint_annotations


def extract_pie_pose_annotations_fixed(
    fig, 
    chart_info_map, 
    cls_map_pose: Dict[str, int], 
    img_w: int, 
    img_h: int
) -> List[Dict]:
    """
    Extract YOLO pose annotations for pie charts (NOVA LÓGICA - 5 PONTOS).
    
    CRITICAL: Annotate each slice (wedge) individually.
    - Classe: 0 ("slice_boundary")
    - Keypoints: 5 (Centro da fatia, Início do Arco, Intermediário 1, Intermediário 2, Fim do Arco)
    
    O centro da fatia respeita a "explosão" (posição deslocada).
    """
    keypoint_annotations = []
    renderer = fig.canvas.get_renderer()
    
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        
        chart_info = chart_info_map.get(ax, {})
        chart_type = chart_info.get('chart_type_str', '')
        
        if chart_type != 'pie':
            continue
        
        pie_geometry = chart_info.get('pie_geometry', None)
        if not pie_geometry:
            continue
        
        wedges_info = pie_geometry.get('wedges', [])
        
        for wedge_geo in wedges_info:
            # 1. Coletar os 5 keypoints em coordenadas de DADOS
            #    'center' é o centro deslocado (se explodido) ou o centro real
            kpt1_data = wedge_geo.get('center')
            kpt2_data = wedge_geo.get('arc_start')
            kpt3_data = wedge_geo.get('arc_end')
            # --- INÍCIO DA MODIFICAÇÃO ---
            kpt4_data = wedge_geo.get('arc_inter_1') # Novo ponto
            kpt5_data = wedge_geo.get('arc_inter_2') # Novo ponto
            
            if not all([kpt1_data, kpt2_data, kpt3_data, kpt4_data, kpt5_data]):
                continue
            # --- FIM DA MODIFICAÇÃO ---
            
            # 2. Transformar os 5 keypoints para coordenadas de PIXEL (com Y invertido)
            kpt1_px_data = ax.transData.transform_point(kpt1_data)
            kpt2_px_data = ax.transData.transform_point(kpt2_data)
            kpt3_px_data = ax.transData.transform_point(kpt3_data)
            # --- INÍCIO DA MODIFICAÇÃO ---
            kpt4_px_data = ax.transData.transform_point(kpt4_data)
            kpt5_px_data = ax.transData.transform_point(kpt5_data)
            
            # Ordem: Centro, Início, Inter 1, Inter 2, Fim
            all_kpts_px = [
                (kpt1_px_data[0], img_h - kpt1_px_data[1]), # 0: Centro
                (kpt2_px_data[0], img_h - kpt2_px_data[1]), # 1: InícioArco
                (kpt4_px_data[0], img_h - kpt4_px_data[1]), # 2: Inter 1
                (kpt5_px_data[0], img_h - kpt5_px_data[1]), # 3: Inter 2
                (kpt3_px_data[0], img_h - kpt3_px_data[1])  # 4: FimArco
            ]
            # --- FIM DA MODIFICAÇÃO ---
            
            # 3. Calcular Bounding Box a partir dos 5 pontos de pixel
            all_x, all_y = zip(*all_kpts_px)
            x0, x1 = min(all_x), max(all_x)
            y0, y1 = min(all_y), max(all_y)
            
            # Normalizar BBox
            cx = max(0.0, min(1.0, (x0 + x1) / 2 / img_w))
            cy = max(0.0, min(1.0, (y0 + y1) / 2 / img_h))
            w = max(0.0, min(1.0, (x1 - x0) / img_w))
            h = max(0.0, min(1.0, (y1 - y0) / img_h))
            
            # 4. Normalizar Keypoints
            kp_norm = []
            for x_px, y_px in all_kpts_px:
                kp_norm.append([
                    max(0.0, min(1.0, x_px / img_w)),
                    max(0.0, min(1.0, y_px / img_h)),
                    2  # Todos visíveis
                ])
            
            # 5. Adicionar anotação
            keypoint_annotations.append({
                'class_id': 0,  # "slice_boundary"
                'bbox': (cx, cy, w, h),
                'keypoints': kp_norm  # Lista de 5 keypoints
            })
    
    return keypoint_annotations
# In generator.py after extract_pie_pose_annotations()

# CORREÇÃO NO ARQUIVO generator.py
# Localização: função extract_line_pose_annotations

def extract_line_pose_annotations_fixed(
    fig, 
    chart_info_map, 
    cls_map_pose: Dict[str, int], 
    img_w: int, 
    img_h: int
) -> List[Dict]:
    """
    Extract YOLO pose annotations for line charts.
    CRITICAL FIX: Use only plotted points for pose construction to preserve sharp features.
    """
    keypoint_annotations = []
    renderer = fig.canvas.get_renderer()
    
    for ax in fig.axes:
        if not ax.get_visible():
            continue
            
        chart_info = chart_info_map.get(ax, {})
        chart_type = chart_info.get('chart_type_str', '')
        
        if chart_type != 'line':
            continue
            
        keypoint_info = chart_info.get('keypoint_info', [])
        if not keypoint_info:
            continue
        
        for series_idx, series_kpts in enumerate(keypoint_info):
            # Use only the plotted points captured after actual drawing
            plotted = series_kpts.get('plotted_points', [])
            if not plotted or len(plotted) < 1:
                continue
            
            # Build 51-point pose using only actual plotted vertices
            kpts = build_51_from_plotted(plotted)
            
            # Transform to pixel coordinates
            px_pts = []
            for x, y, _ in kpts:
                px, py = ax.transData.transform_point((x, y))
                px_pts.append((px, img_h - py))  # Y-flip
            
            # Calculate bounding box from the 51 pixel points
            xs, ys = zip(*px_pts)
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            
            # Normalize bbox
            cx = max(0.0, min(1.0, (x0 + x1) / 2 / img_w))
            cy = max(0.0, min(1.0, (y0 + y1) / 2 / img_h))
            w = max(0.0, min(1.0, (x1 - x0) / img_w))
            h = max(0.0, min(1.0, (y1 - y0) / img_h))
            
            # Normalize keypoints
            kp_norm = []
            for x, y in px_pts:
                kp_norm.append([
                    max(0.0, min(1.0, x / img_w)),
                    max(0.0, min(1.0, y / img_h)),
                    2  # All visible
                ])
            
            if GENERATION_CONFIG.get('debug_coords', False):
                print(f"DEBUG [LINE-POSE] Series {series_idx}: Built 51 from {len(plotted)} plotted points")
                if len(plotted) > 0:
                    print(f"DEBUG [LINE-POSE] Series {series_idx}: Original x range: [{plotted[0][0]:.2f}, {plotted[-1][0]:.2f}]")
                if len(kp_norm) > 0:
                    x_norms = [kp[0] for kp in kp_norm]
                    print(f"DEBUG [LINE-POSE] Series {series_idx}: Final x range: [{min(x_norms):.4f}, {max(x_norms):.4f}]")
                    # Check monotonicity
                    is_monotonic = all(x_norms[i] <= x_norms[i+1] for i in range(len(x_norms)-1))
                    print(f"DEBUG [LINE-POSE] Series {series_idx}: X coordinates monotonic: {is_monotonic}")
            
            keypoint_annotations.append({
                'class_id': 0,  # line_boundary
                'bbox': (cx, cy, w, h),
                'keypoints': kp_norm
            })
    
    return keypoint_annotations


def apply_realism_effects(pil_img, annotations, effects_config):
    """Apply realism effects and return modified image and annotations."""
    effect_function_map = {
        "blur": apply_blur_effect, 
        "motion_blur": apply_motion_blur_effect,
        "low_res": apply_low_res_effect, 
        "noise": apply_noise_effect,
        "jpeg_compression": apply_jpeg_compression_effect, 
        "pixelation": apply_pixelation_effect,
        "posterize": apply_posterize_effect, 
        "color_variation": apply_color_variation_effect,
        "ui_chrome": apply_ui_chrome_effect, 
        "watermark": apply_watermark_effect,
        "vignette": apply_vignette_effect, 
        "scanner_streaks": apply_scanner_streaks_effect,
        "clipping": apply_clipping_effect, 
        "printing_artifacts": apply_printing_artifacts_effect,
        "mouse_cursor": apply_mouse_cursor_effect, 
        "text_degradation": apply_text_degradation_effect,
        "grid_occlusion": apply_grid_occlusion_effect, 
        "scan_rotation": apply_scan_rotation_effect,
        "grayscale": apply_grayscale_effect, 
        "perspective": apply_perspective_effect,
    }
    
    total_dx, total_dy = 0, 0
    
    for effect_name, effect_config in effects_config.items():
        if random.random() < effect_config.get('p', 0):
            func = effect_function_map.get(effect_name)
            if not func: 
                continue
            
            print(f"    - Applying effect: {effect_name}")
            params = effect_config.get('params', {})
            
            try:
                if effect_name == 'clipping':
                    pil_img, dx, dy = func(pil_img, **params)
                    total_dx += dx
                    total_dy += dy
                elif effect_name == 'scan_rotation':
                    result = func(pil_img, **params)
                    pil_img = result[0]
                else:
                    pil_img = func(pil_img, **params)
            except Exception as e:
                print(f"      [WARNING] Failed to apply effect '{effect_name}': {e}")
    
    # Apply offset to annotations
    if total_dx != 0 or total_dy != 0:
        print(f"    - Applying total annotation offset: dx={total_dx}, dy={total_dy}")
        for ann in annotations:
            bbox = ann['bbox']
            new_bbox = transforms.Bbox.from_extents(
                bbox.x0 + total_dx, bbox.y0 + total_dy,
                bbox.x1 + total_dx, bbox.y1 + total_dy
            )
            ann['bbox'] = new_bbox
    
    return pil_img, annotations

def save_annotations_yolo(annotations, img_w, img_h, output_path):
    """Save in proper YOLO format with normalization"""
    with open(output_path, 'w') as f:
        for ann in annotations:
            class_id = ann['class_id']
            bbox = ann['bbox']
            
            # Extract bbox coordinates
            if hasattr(bbox, 'extents'):
                x0, y0, x1, y1 = bbox.extents
            else:
                x0, y0, x1, y1 = bbox
            
            # Convert matplotlib (bottom-left) to image (top-left) coordinates
            img_y0 = img_h - y1  # Top edge
            img_y1 = img_h - y0  # Bottom edge
            
            # Clamp to bounds
            img_x0 = max(0.0, min(float(x0), img_w))
            img_x1 = max(img_x0, min(float(x1), img_w))
            img_y0 = max(0.0, min(float(img_y0), img_h))
            img_y1 = max(img_y0, min(float(img_y1), img_h))
            
            # YOLO format: normalized center and dimensions
            x_center = (img_x0 + img_x1) / 2.0 / img_w
            y_center = (img_y0 + img_y1) / 2.0 / img_h
            width = (img_x1 - img_x0) / img_w
            height = (img_y1 - img_y0) / img_h
            
            # Clamp to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))
            
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# In generator.py after save_annotations_yolo()
def save_annotations_pose_fixed(
    annotations: List[Dict], 
    img_w: int, 
    img_h: int, 
    output_path: str
):
    """
    Save YOLO pose format annotations to file.
    
    Format per line:
    class_id center_x center_y width height kpt1_x kpt1_y vis1 kpt2_x kpt2_y vis2 ...
    """
    with open(output_path, 'w') as f:
        for i, ann in enumerate(annotations):
            class_id = ann['class_id']
            cx, cy, w, h = ann['bbox']
            keypoints = ann['keypoints']
            
            # Format: class bbox kpt1 kpt2 ... kptn
            line_parts = [str(class_id), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
            
            for x, y, vis in keypoints:
                line_parts.extend([f"{x:.6f}", f"{y:.6f}", str(vis)])
            
            line_str = " ".join(line_parts)
            f.write(line_str + "\n")
            
            # Enhanced debug logging - output the entire row for verification
            if GENERATION_CONFIG.get('debug_coords', False):
                print(f"Saving annotation {i}: {line_str}")
                
                # Add complete pose format display
                print(f"DEBUG [POSE-FORMAT] Annotation {i}: Class={class_id}, BBox=({cx:.4f},{cy:.4f},{w:.4f},{h:.4f})")
                print(f"DEBUG [POSE-FORMAT] Annotation {i}: Total keypoints={len(keypoints)}")
                
                # Show keypoints grouped by category for better visualization
                if len(keypoints) >= 51:
                    # Start point (index 0)
                    start_x, start_y, start_vis = keypoints[0]
                    print(f"DEBUG [POSE-FORMAT] Annotation {i}: Start point: ({start_x:.4f},{start_y:.4f},vis={start_vis})")
                    
                    # Boundary points (indices 1-25)
                    boundary_points = keypoints[1:26]
                    visible_boundary = [(j+1, x, y, vis) for j, (x, y, vis) in enumerate(boundary_points) if vis > 0]
                    print(f"DEBUG [POSE-FORMAT] Annotation {i}: Boundary points: {len(visible_boundary)} visible")
                    if visible_boundary:
                        # Show first 3 and last 3 boundary points
                        first_3 = visible_boundary[:3]
                        last_3 = visible_boundary[-3:] if len(visible_boundary) > 3 else visible_boundary[3:]
                        for j, x, y, vis in first_3:
                            print(f"DEBUG [POSE-FORMAT] Annotation {i}:   Boundary {j}: ({x:.4f},{y:.4f},vis={vis})")
                        if len(visible_boundary) > 6:
                            print(f"DEBUG [POSE-FORMAT] Annotation {i}:   ... ({len(visible_boundary)-6} more points) ...")
                        for j, x, y, vis in last_3:
                            print(f"DEBUG [POSE-FORMAT] Annotation {i}:   Boundary {j}: ({x:.4f},{y:.4f},vis={vis})")
                    
                    # Additional keypoints (indices 26-50)
                    remaining_kpts = keypoints[26:]
                    visible_remaining = [(j+26, x, y, vis) for j, (x, y, vis) in enumerate(remaining_kpts) if vis > 0]
                    if visible_remaining:
                        print(f"DEBUG [POSE-FORMAT] Annotation {i}: Additional points: {len(visible_remaining)} visible")
                        for j, x, y, vis in visible_remaining[:5]:  # Show first 5
                            print(f"DEBUG [POSE-FORMAT] Annotation {i}:   Point {j}: ({x:.4f},{y:.4f},vis={vis})")
                
                # --- INÍCIO DA MODIFICAÇÃO (Bloco 1) ---
                # Handle Pie Slices
                elif len(keypoints) == 5: # Handle Pie Slices
                    print(f"DEBUG [POSE-FORMAT] Annotation {i}: PIE SLICE (5 keypoints)")
                    kpt_names = ['WedgeCenter', 'ArcStart', 'ArcInter1', 'ArcInter2', 'ArcEnd']
                    for j, kpt in enumerate(keypoints):
                        x, y, vis = kpt
                        print(f"DEBUG [POSE-FORMAT] Annotation {i}:   {kpt_names[j]}: ({x:.4f},{y:.4f},vis={vis})")
                # --- FIM DA MODIFICAÇÃO (Bloco 1) ---
                
                # Add sequential coordinate logging to verify path following
                if len(keypoints) >= 25:  # At least boundary points
                    # Extract the boundary keypoints (indices 1-25) to verify path sequence
                    boundary_kpts = keypoints[1:26]  # Keypoints 1-25 are boundary points
                    visible_boundary = [(j+1, x, y, vis) for j, (x, y, vis) in enumerate(boundary_kpts) if vis > 0]
                    
                    if visible_boundary:
                        # Log first few coordinates in sequence to verify path following
                        coords_sequence = [(x, y) for j, x, y, vis in visible_boundary[:10]]
                        print(f"DEBUG [PATH-SEQUENCE] Annotation {i}: First 10 boundary coordinates: {coords_sequence}")
                        
                        # Check if coordinates follow increasing X (typical for charts)
                        if len(coords_sequence) > 1:
                            x_values = [x for x, y in coords_sequence]
                            is_x_increasing = all(x_values[k] <= x_values[k+1] for k in range(len(x_values)-1))
                            print(f"DEBUG [PATH-ORDER] Annotation {i}: X coordinates monotonically increasing: {is_x_increasing}")
                        
                        # Add index progression verification
                        print(f"DEBUG [INDEX-PROGRESSION] Annotation {i}: Visible boundary point indices: {[j for j, x, y, vis in visible_boundary[:10]]}")
                        
                        # Verify that indices are in proper sequence (should generally be increasing)
                        indices = [j for j, x, y, vis in visible_boundary]
                        if len(indices) > 1:
                            index_differences = [indices[k+1] - indices[k] for k in range(len(indices)-1)]
                            avg_diff = sum(index_differences) / len(index_differences) if index_differences else 0
                            print(f"DEBUG [INDEX-PROGRESSION] Annotation {i}: Average index step: {avg_diff:.2f}")
            
            # Enhanced debug output for coordinate analysis
            if GENERATION_CONFIG.get('debug_coords', False):
                # Also write to a debug file with more details
                debug_path = output_path.replace('.txt', '_debug.txt')
                with open(debug_path, 'a') as debug_f:
                    debug_f.write(f"Annotation {i} (Class {class_id}):\n")
                    debug_f.write(f"  BBox: ({cx:.4f}, {cy:.4f}, {w:.4f}, {h:.4f})\n")
                    debug_f.write(f"  Complete row: {line_str}\n")
                    debug_f.write(f"  Keypoints: {len(keypoints)} total\n")
                    
                    # Log first few and last few keypoints to verify ordering
                    visible_kpts = [(j, x, y, vis) for j, (x, y, vis) in enumerate(keypoints) if vis > 0]
                    if visible_kpts:
                        debug_f.write(f"  Visible keypoints: {len(visible_kpts)} of {len(keypoints)}\n")
                        
                        # --- INÍCIO DA MODIFICAÇÃO (Bloco 2) ---
                        if len(keypoints) == 5: # PIE SLICE DEBUG
                            # Map the original index (0-4) to the correct name
                            kpt_name_map = {
                                0: 'WedgeCenter', 
                                1: 'ArcStart', 
                                2: 'ArcInter1', 
                                3: 'ArcInter2', 
                                4: 'ArcEnd'
                            }
                            for (idx, x, y, vis) in visible_kpts:
                                name = kpt_name_map.get(idx, f"Kpt {idx}")
                                debug_f.write(f"    {name}: ({x:.4f}, {y:.4f}, vis={vis})\n")
                        # --- FIM DA MODIFICAÇÃO (Bloco 2) ---
                                
                        elif len(visible_kpts) >= 6:  # Show first 3 and last 3 if many keypoints
                            for j, x, y, vis in visible_kpts[:3]:
                                debug_f.write(f"    Kpt {j}: ({x:.4f}, {y:.4f}, vis={vis})\n")
                            debug_f.write(f"    ... ({len(visible_kpts)-6} more keypoints) ...\n")
                            for j, x, y, vis in visible_kpts[-3:]:
                                debug_f.write(f"    Kpt {j}: ({x:.4f}, {y:.4f}, vis={vis})\n")
                        else:  # Show all if few keypoints
                            for j, x, y, vis in visible_kpts:
                                debug_f.write(f"    Kpt {j}: ({x:.4f}, {y:.4f}, vis={vis})\n")
                    else:
                        debug_f.write(f"  No visible keypoints\n")
                    
                    # Check and log normalization status
                    all_normalized = all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y, vis in keypoints)
                    debug_f.write(f"  All coordinates normalized [0,1]: {all_normalized}\n")
                    
                    # Add comprehensive normalization verification
                    if GENERATION_CONFIG.get('debug_coords', False):
                        # Check each coordinate individually for detailed reporting
                        out_of_range_coords = []
                        for j, (x, y, vis) in enumerate(keypoints):
                            if vis > 0:  # Only check visible keypoints
                                if not (0.0 <= x <= 1.0):
                                    out_of_range_coords.append((j, 'x', x))
                                if not (0.0 <= y <= 1.0):
                                    out_of_range_coords.append((j, 'y', y))
                        
                        if out_of_range_coords:
                            print(f"DEBUG [NORMALIZATION-ERROR] Annotation {i}: Out of range coordinates found:")
                            for coord_idx, coord_type, value in out_of_range_coords[:5]:  # Show first 5 errors
                                print(f"  Keypoint {coord_idx}: {coord_type} = {value:.6f} (should be [0,1])")
                        else:
                            # Add detailed normalization verification
                            x_values = [x for x, y, vis in keypoints if vis > 0]
                            y_values = [y for x, y, vis in keypoints if vis > 0]
                            if x_values and y_values:
                                x_min, x_max = min(x_values), max(x_values)
                                y_min, y_max = min(y_values), max(y_values)
                                print(f"DEBUG [NORMALIZATION] Annotation {i}: X range [{x_min:.6f}, {x_max:.6f}], Y range [{y_min:.6f}, {y_max:.6f}]")
                    
                    debug_f.write("\n")
            
            # Add enhanced pose format verification
            if GENERATION_CONFIG.get('debug_coords', False):
                # Create a temporary annotation for verification
                temp_annotation = {'class_id': class_id, 'bbox': (cx, cy, w, h), 'keypoints': keypoints}
                verify_pose_format([temp_annotation], f"SAVING-ANNOTATION-{i}")



def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle matplotlib Bbox objects
    elif hasattr(obj, 'extents'):  # This is a matplotlib Bbox
        x0, y0, x1, y1 = obj.extents
        return [float(x0), float(y0), float(x1), float(y1)]
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj

def get_detailed_annotations(fig, chart_info_map, cls_map, img_w, img_h, raw_annotations=None):
    """Extract comprehensive metadata with xyxy coordinates and text content
    
    Args:
        raw_annotations: Optional list of raw annotation dicts from get_granular_annotations
    """
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.get_window_extent(renderer)
    seen = set()
    
    detailed_metadata = {
        "chart_type": None,
        "orientation": None,
        "scale_labels": [],
        "tick_labels": [],
        "chart_title": [],
        "axis_title": [],
        "legend": [],
        "bar": [],
        "data_point": [],
        "error_bar": [],
        "significance_marker": [],
        "data_label": [],
        "box": [],
        "median_line": [],
        "range_indicator": [],
        "outlier": []
    }
    
    def add_annotation(element_type, bbox, text="", conf=1.0, extra=None):
        if not bbox or bbox.width <= 1 or bbox.height <= 1:
            return None
        
        key = (element_type, round(bbox.x0, 2), round(bbox.y0, 2), round(bbox.x1, 2), round(bbox.y1, 2))
        if key in seen:
            return None
        seen.add(key)
        
        xyxy = bbox_to_xyxy(bbox, img_h)
        entry = {"xyxy": xyxy, "conf": conf}
        if text:
            entry["text"] = text
        if extra:
            entry.update(extra)
        
        detailed_metadata[element_type].append(entry)
        return entry
    
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        
        chart_info = chart_info_map.get(ax, {})
        chart_type = chart_info.get('chart_type_str', 'unknown')
        orientation = chart_info.get('orientation', 'vertical')
        
        detailed_metadata["chart_type"] = chart_type
        detailed_metadata["orientation"] = orientation
        
        # Chart Title
        if 'chart_title' in cls_map:
            title = ax.title
            if title and title.get_visible() and title.get_text():
                add_annotation("chart_title", title.get_window_extent(renderer), 
                             text=title.get_text().strip(), conf=1.0)
        
        # Axis Titles
        if 'axis_title' in cls_map:
            if ax.xaxis.label.get_visible() and ax.xaxis.label.get_text():
                add_annotation("axis_title", ax.xaxis.label.get_window_extent(renderer),
                             text=ax.xaxis.label.get_text().strip(), conf=1.0, 
                             extra={"axis": "x"})
            if ax.yaxis.label.get_visible() and ax.yaxis.label.get_text():
                add_annotation("axis_title", ax.yaxis.label.get_window_extent(renderer),
                             text=ax.yaxis.label.get_text().strip(), conf=1.0,
                             extra={"axis": "y"})
        
        # Scale Labels (Tick Labels)
        if 'axis_labels' in cls_map:
            # Use the scale axis information from the chart generation
            scale_axis_info = chart_info.get('scale_axis_info', {})
            primary_scale_axis = scale_axis_info.get('primary_scale_axis', 'y')
            secondary_scale_axis = scale_axis_info.get('secondary_scale_axis', None)
            bg_color = ax.get_facecolor()
            # Process X-axis labels
            for label in ax.get_xticklabels():
                if label.get_visible() and label.get_text():
                    if has_non_background_pixels(label, fig, ax, bg_color, threshold=5):
                        txt = label.get_text().strip()
                        # Check if X-axis is a scale axis
                        if is_float(txt) and (primary_scale_axis == 'x' or secondary_scale_axis == 'x'):
                            add_annotation("scale_labels", label.get_window_extent(renderer),
                                        text=txt, conf=1.0, extra={"axis": "x", "is_numeric": True})
                        else:
                            add_annotation("tick_labels", label.get_window_extent(renderer),
                                        text=txt, conf=1.0, extra={"axis": "x", "is_numeric": is_float(txt)})
                    else:
                        if GENERATION_CONFIG.get('debug_mode', False):
                            print(f"DEBUG: label '{label.get_text()}' empty (no pixels), skipping")
            
            # Process Y-axis labels
            for label in ax.get_yticklabels():
                if label.get_visible() and label.get_text():
                    if has_non_background_pixels(label, fig, ax, bg_color, threshold=5):
                        txt = label.get_text().strip()
                        # Check if Y-axis is a scale axis
                        if is_float(txt) and (primary_scale_axis == 'y' or secondary_scale_axis == 'y'):
                            add_annotation("scale_labels", label.get_window_extent(renderer),
                                        text=txt, conf=1.0, extra={"axis": "y", "is_numeric": True})
                        else:
                            add_annotation("tick_labels", label.get_window_extent(renderer),
                                        text=txt, conf=1.0, extra={"axis": "y", "is_numeric": is_float(txt)})
                    else:
                        if GENERATION_CONFIG.get('debug_mode', False):
                            print(f"DEBUG: label '{label.get_text()}' empty (no pixels), skipping")
                
        # Legend
        if 'legend' in cls_map:
            legend = ax.get_legend()
            if legend and legend.get_visible():
                # Check if the legend has non-background pixels before adding annotation
                if has_non_background_pixels(legend, fig, ax, ax.get_facecolor(), threshold=5):
                    legend_texts = [t.get_text() for t in legend.get_texts() if t.get_visible()]
                    add_annotation("legend", legend.get_window_extent(renderer),
                                 conf=1.0, extra={"entries": legend_texts})
                else:
                    if GENERATION_CONFIG.get('debug_mode', False):
                        print(f"DEBUG: Legend empty (no pixels), skipping")
        
        # Data Elements
        for artist in chart_info.get('data_artists', []):
            if not artist or not artist.get_visible():
                continue
            
            # Line chart points
            if isinstance(artist, matplotlib.lines.Line2D) and chart_type == 'line':
                x_data, y_data = artist.get_xdata(), artist.get_ydata()
                for x, y in zip(x_data, y_data):
                    px, py = ax.transData.transform_point((x, y))
                    bbox = transforms.Bbox.from_extents(px - 5, py - 5, px + 5, py + 5)
                    add_annotation("data_point", bbox, conf=0.95, 
                                 extra={"x": float(x), "y": float(y)})
            
            # Scatter points
            elif 'PathCollection' in str(type(artist)) and chart_type == 'scatter':
                offsets = artist.get_offsets()
                sizes = artist.get_sizes()
                if len(offsets):
                    is_uniform = (sizes.size == 1)
                    pts_to_px = fig.dpi / 72.0
                    for i, (x, y) in enumerate(offsets):
                        px, py = ax.transData.transform_point((x, y))
                        s = sizes[0] if is_uniform else sizes[i]
                        r = np.sqrt(s / np.pi) * pts_to_px
                        bbox = transforms.Bbox.from_extents(px - r, py - r, px + r, py + r)
                        add_annotation("data_point", bbox, conf=0.95,
                                     extra={"x": float(x), "y": float(y), "size": float(s)})
            
            # Bar elements
            elif isinstance(artist, patches.Rectangle) and chart_type == 'bar':
                value = float(artist.get_height() if orientation == 'vertical' else artist.get_width())
                add_annotation("bar", artist.get_window_extent(renderer), conf=0.98,
                             extra={"value": value})
        
        # Box plot elements
        if chart_type == 'box':
            bp_dict = chart_info.get('boxplot_dict', {})
            if bp_dict:
                for box in bp_dict.get('boxes', []):
                    add_annotation("box", box.get_window_extent(renderer), conf=0.98)
                
                for median in bp_dict.get('medians', []):
                    bbox = median.get_window_extent(renderer)
                    padded = transforms.Bbox.from_extents(bbox.x0, bbox.y0 - 2, bbox.x1, bbox.y1 + 2)
                    add_annotation("median_line", padded, conf=0.97)
                
                for i in range(len(bp_dict.get('boxes', []))):
                    try:
                        w1 = bp_dict['whiskers'][2 * i]
                        w2 = bp_dict['whiskers'][2 * i + 1]
                        c1 = bp_dict['caps'][2 * i]
                        c2 = bp_dict['caps'][2 * i + 1]
                        combined = transforms.Bbox.union([
                            w1.get_window_extent(renderer),
                            w2.get_window_extent(renderer),
                            c1.get_window_extent(renderer),
                            c2.get_window_extent(renderer)
                        ])
                        add_annotation("range_indicator", combined, conf=0.96)
                    except IndexError:
                        continue
                
                for flier in bp_dict.get('fliers', []):
                    for x, y in zip(flier.get_xdata(), flier.get_ydata()):
                        px, py = ax.transData.transform_point((x, y))
                        bbox = transforms.Bbox.from_extents(px - 3, py - 3, px + 3, py + 3)
                        add_annotation("outlier", bbox, conf=0.94)
        
        # Error bars and text annotations
        for artist in chart_info.get('other_artists', []):
            if not artist:
                continue
            
            # Error bars
            if isinstance(artist, ErrorbarContainer):
                plotline, caplines, barlinecols = artist.lines
                if barlinecols and caplines:
                    stems = barlinecols[0].get_segments()
                    for i in range(min(len(stems), len(caplines) // 2)):
                        try:
                            p1 = ax.transData.transform(stems[i][0])
                            p2 = ax.transData.transform(stems[i][1])
                            stem_bbox = transforms.Bbox([p1, p2])
                            
                            cap1, cap2 = caplines[2 * i], caplines[2 * i + 1]
                            c1x, c1y = cap1.get_data()
                            c2x, c2y = cap2.get_data()
                            cap1_bbox = transforms.Bbox([
                                ax.transData.transform((c1x[0], c1y[0])),
                                ax.transData.transform((c1x[1], c1y[1]))
                            ])
                            cap2_bbox = transforms.Bbox([
                                ax.transData.transform((c2x[0], c2y[0])),
                                ax.transData.transform((c2x[1], c2y[1]))
                            ])
                            
                            final = transforms.Bbox.union([stem_bbox, cap1_bbox, cap2_bbox])
                            add_annotation("error_bar", final, conf=0.89)
                        except Exception:
                            continue
            
            # Text annotations
            elif isinstance(artist, matplotlib.text.Text) and artist.get_visible():
                txt = artist.get_text().strip()
                if not txt:
                    continue
                
                # Significance markers
                if txt in ['*', '**', '***', 'ns', 'a', 'b', 'c', 'd']:
                    add_annotation("significance_marker", artist.get_window_extent(renderer),
                                 text=txt, conf=0.92)
                # Data labels
                elif is_float(txt.replace('%', '')):
                    add_annotation("data_label", artist.get_window_extent(renderer),
                                 text=txt, conf=0.91)
    
    # CRITICAL FIX: Add raw XYXY coordinates at the end
    if raw_annotations:
        detailed_metadata["raw_annotations"] = []
        for ann in raw_annotations:
            bbox = ann['bbox']
            xyxy = bbox_to_xyxy_absolute(bbox, img_h)
            detailed_metadata["raw_annotations"].append({
                'class_id': int(ann['class_id']),
                'xyxy': [int(coord) for coord in xyxy]
            })
    
    return detailed_metadata

def create_unified_annotation(fig, chart_info_map, cls_map, img_w, img_h, annotations):
    """
    Create comprehensive unified JSON with complete chart generation metadata.
    This replaces the existing createunifiedannotation function.
    """
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.get_window_extent(renderer)

    # Build reverse lookup map
    id_to_name = {v: k for k, v in cls_map.items()}

    detailed_metadata = {
        "chart_type": None,
        "orientation": None,
        "scale_labels": [],
        "tick_labels": [],
        "chart_title": [],
        "axis_title": [],
        "legend": [],
        "bar": [],
        "datapoint": [],
        "errorbar": [],
        "significancemarker": [],
        "datalabel": [],
        "box": [],
        "medianline": [],
        "rangeindicator": [],
        "outlier": [],
        "wedge": [],
        "linesegment": [],
        "areaboundary": [],
        "cell": [],
        "colorbar": [],
        "connectorline": [],

        # NEW: Chart generation metadata
        "scale_axis_info": {},
        "bar_info": [],
        "keypoint_info": [],
        "boxplot_metadata": {},
        "pie_geometry": {},
        "series_count": 1,
        "series_names": [],
        "stacking_mode": None,
        "dual_axis_info": {},
        "style": None,
        "pattern": None,
        "is_scientific": False
    }

    # Extract chart metadata from chart_info_map
    for ax in fig.axes:
        if not ax.get_visible():
            continue

        chart_info = chart_info_map.get(ax, {})

        # Basic chart information
        chart_type = chart_info.get("chart_type_str", "unknown")
        detailed_metadata["chart_type"] = chart_type
        detailed_metadata["orientation"] = chart_info.get("orientation", "vertical")

        # NEW: Extract scale axis information
        from chart import extract_scale_axis_info
        scale_axis_info = extract_scale_axis_info(ax, chart_type)
        if scale_axis_info:
            detailed_metadata["scale_axis_info"] = scale_axis_info

        # NEW: Extract bar info (for bar and histogram charts)
        from chart import extract_bar_info
        bar_info_list = extract_bar_info(ax, chart_type)
        if bar_info_list:
            detailed_metadata["bar_info"] = [
                {
                    "center": float(info.get("center", 0)),
                    "height": float(info.get("height", 0)),
                    "width": float(info.get("width", 0)),
                    "bottom": float(info.get("bottom", 0)),
                    "top": float(info.get("top", 0)),
                    "series_idx": info.get("series_idx"),
                    "bar_idx": info.get("bar_idx"),
                    "axis": info.get("axis", "primary")
                }
                for info in bar_info_list
            ]

        # NEW: Extract keypoint info (for line, area, pie charts)
        from chart import extract_keypoint_info
        keypoint_info = extract_keypoint_info(ax, chart_type)
        if keypoint_info:
            detailed_metadata["keypoint_info"] = [
                {
                    "series_idx": kp.get("series_idx"),
                    "points": [
                        {
                            "x": float(pt.get("x", 0)),
                            "y": float(pt.get("y", 0)),
                            "is_inflection": pt.get("is_inflection", False)
                        }
                        for pt in kp.get("points", [])
                    ]
                }
                for kp in keypoint_info
            ]

        # FIX: boxplot_dict now contains the pre-processed boxplot_metadata 
        # from _generate_boxplot_chart (with 'medians' as a list of dicts)
        boxplot_dict = chart_info.get('boxplot_dict', {})
        if boxplot_dict and chart_type == 'box':
            # boxplot_dict already contains the correct structure:
            # {'num_groups': N, 'box_width': W, 'orientation': 'vertical', 'medians': [...]}
            detailed_metadata["boxplot_metadata"] = {
                "num_groups": boxplot_dict.get("num_groups", 0),
                "box_width": float(boxplot_dict.get("box_width", 0)),
                "orientation": boxplot_dict.get("orientation", "vertical"),
                "medians": [
                    {
                        "group_index": m.get("group_index"),
                        "group_label": m.get("group_label"),
                        "median_value": float(m.get("median_value", 0)),
                        "lower_left": m.get("lower_left", {}),
                        "upper_right": m.get("upper_right", {}),
                        "center_x": m.get("center_x"),
                        "center_y": m.get("center_y"),
                        "line_length": float(m.get("line_length", 0))
                    }
                    for m in boxplot_dict.get("medians", [])
                ]
            }

        # NEW: Extract pie geometry
        from chart import extract_pie_geometry
        pie_geometry = extract_pie_geometry(ax, chart_type)
        if pie_geometry:
            detailed_metadata["pie_geometry"] = {
                "center_point": {
                    "x": float(pie_geometry.get("center_point", {}).get("x", 0)),
                    "y": float(pie_geometry.get("center_point", {}).get("y", 0))
                },
                "radius": float(pie_geometry.get("radius", 0)),
                "wedges": [
                    {
                        "wedge_index": w.get("wedge_index"),
                        "start_angle": float(w.get("start_angle", 0)),
                        "end_angle": float(w.get("end_angle", 0)),
                        "mid_angle": float(w.get("mid_angle", 0)),
                        "percentage": float(w.get("percentage", 0))
                    }
                    for w in pie_geometry.get("wedges", [])
                ]
            }

        # NEW: Extract series information
        detailed_metadata["series_count"] = chart_info.get("series_count", 1)
        detailed_metadata["series_names"] = chart_info.get("series_names", [])
        detailed_metadata["stacking_mode"] = chart_info.get("stacking_mode")
        detailed_metadata["dual_axis_info"] = chart_info.get("dual_axis_info", {})

        # NEW: Extract style information
        detailed_metadata["style"] = chart_info.get("style")
        detailed_metadata["pattern"] = chart_info.get("pattern")
        detailed_metadata["is_scientific"] = chart_info.get("is_scientific", False)

    # Extract text content from all text artists
    text_lookup = {}

    for ax in fig.axes:
        if not ax.get_visible():
            continue

        chart_info = chart_info_map.get(ax, {})

        # Process all text artists
        for artist in chart_info.get('other_artists', []):
            if hasattr(artist, 'get_text') and artist.get_visible():
                try:
                    text_content = artist.get_text().strip()
                    if text_content:
                        bbox = artist.get_window_extent(renderer)
                        # CRITICAL FIX: More precise bbox key generation
                        key = (round(bbox.x0, 1), round(bbox.y0, 1), round(bbox.x1, 1), round(bbox.y1, 1))
                        text_lookup[key] = text_content

                        if GENERATION_CONFIG.get('debug_mode', False) or GENERATION_CONFIG.get('debug_annotations', False):
                            print(f"DEBUG: Text lookup added: '{text_content}' at {key}")
                except Exception as e:
                    if GENERATION_CONFIG.get('debug_mode', False) or GENERATION_CONFIG.get('debug_annotations', False):
                        print(f"DEBUG: Text extraction failed: {e}")

    if GENERATION_CONFIG.get('debug_mode', False) or GENERATION_CONFIG.get('debug_annotations', False):
        print(f"DEBUG: create_unified_annotation - Processing {len(annotations)} annotations")
        print(f"DEBUG: create_unified_annotation - Text lookup contains {len(text_lookup)} entries")

    # Process each annotation
    seen = set()
    def add_annotation(element_type, bbox, text="", conf=1.0, extra=None):
        if not bbox or bbox.width < 1 or bbox.height < 1:
            return None

        key = (element_type, round(bbox.x0, 2), round(bbox.y0, 2),
               round(bbox.x1, 2), round(bbox.y1, 2))
        if key in seen:
            return None
        seen.add(key)

        xyxy = bbox_to_xyxy(bbox, img_h)
        entry = {"xyxy": xyxy, "conf": conf}
        if text:
            entry["text"] = text
        if extra:
            entry.update(extra)

        detailed_metadata[element_type].append(entry)
        return entry

    # Extract annotations from matplotlib objects
    for ax in fig.axes:
        if not ax.get_visible():
            continue

        chart_info = chart_info_map.get(ax, {})
        chart_type = chart_info.get("chart_type_str", "unknown")

        # Chart Title
        if "charttitle" in cls_map:
            title = ax.title
            if title and title.get_visible() and title.get_text():
                add_annotation("charttitle", title.get_window_extent(renderer),
                             text=title.get_text().strip(), conf=1.0)

        # Axis Titles
        if "axistitle" in cls_map:
            if ax.xaxis.label.get_visible() and ax.xaxis.label.get_text():
                add_annotation("axistitle", ax.xaxis.label.get_window_extent(renderer),
                             text=ax.xaxis.label.get_text().strip(), conf=1.0,
                             extra={"axis": "x"})
            if ax.yaxis.label.get_visible() and ax.yaxis.label.get_text():
                add_annotation("axistitle", ax.yaxis.label.get_window_extent(renderer),
                             text=ax.yaxis.label.get_text().strip(), conf=1.0,
                             extra={"axis": "y"})

        # Scale and Tick Labels
        if "axislabels" in cls_map:
            scale_axis_info = detailed_metadata.get("scale_axis_info", {})
            primary_scale_axis = scale_axis_info.get("primary_scale_axis", "y")
            secondary_scale_axis = scale_axis_info.get("secondary_scale_axis", None)
            bgcolor = ax.get_facecolor()

            for label in ax.get_xticklabels():
                if label.get_visible() and label.get_text():
                    if has_non_background_pixels(label, fig, ax, bgcolor, threshold=5):
                        txt = label.get_text().strip()
                        if is_float(txt) and (primary_scale_axis == "x" or secondary_scale_axis == "x"):
                            add_annotation("scalelabels", label.get_window_extent(renderer),
                                         text=txt, conf=1.0,
                                         extra={"axis": "x", "is_numeric": True})
                        else:
                            add_annotation("ticklabels", label.get_window_extent(renderer),
                                         text=txt, conf=1.0,
                                         extra={"axis": "x", "is_numeric": is_float(txt)})

            for label in ax.get_yticklabels():
                if label.get_visible() and label.get_text():
                    if has_non_background_pixels(label, fig, ax, bgcolor, threshold=5):
                        txt = label.get_text().strip()
                        if is_float(txt) and (primary_scale_axis == "y" or secondary_scale_axis == "y"):
                            add_annotation("scalelabels", label.get_window_extent(renderer),
                                         text=txt, conf=1.0,
                                         extra={"axis": "y", "is_numeric": True})
                        else:
                            add_annotation("ticklabels", label.get_window_extent(renderer),
                                         text=txt, conf=1.0,
                                         extra={"axis": "y", "is_numeric": is_float(txt)})

    # Add raw annotations
    detailed_metadata["raw_annotations"] = annotations

    return detailed_metadata

class HeatmapQualityValidator:
    """Comprehensive quality checks for generated heatmaps."""
    
    def __init__(self, config):
        self.config = config
        self.failures = []
    
    def validate_data_structure(self, data):
        """Check for realistic spatial patterns using Moran's I for spatial autocorrelation."""
        try:
            # Calculate spatial autocorrelation using a simplified approach
            # Moran's I measures spatial autocorrelation (values near 0 = random, > 0 = clustered)
            rows, cols = data.shape
            
            if rows < 2 or cols < 2:
                return True  # Skip validation for very small matrices
            
            # Calculate row and column neighbors (simplified adjacency)
            # For a 2D grid, we look at immediate horizontal and vertical neighbors
            neighbor_sums = 0
            neighbor_count = 0
            
            # Calculate mean
            mean_val = data.mean()
            
            # Calculate neighbor relationships for Moran's I (simplified)
            for i in range(rows):
                for j in range(cols):
                    current_val = data[i, j]
                    
                    # Check neighbors (up, down, left, right)
                    neighbors = []
                    if i > 0: neighbors.append(data[i-1, j])  # up
                    if i < rows-1: neighbors.append(data[i+1, j])  # down
                    if j > 0: neighbors.append(data[i, j-1])  # left
                    if j < cols-1: neighbors.append(data[i, j+1])  # right
                    
                    for neighbor_val in neighbors:
                        neighbor_sums += (current_val - mean_val) * (neighbor_val - mean_val)
                        neighbor_count += 1
            
            if neighbor_count == 0:
                return True  # No neighbors (single cell)
                
            # Calculate variance
            variance = np.var(data)
            
            if variance == 0:
                # All values are the same - not spatially interesting but valid
                return True
            
            # Simplified Moran's I calculation
            moran_i = (neighbor_count / (rows * cols)) * (neighbor_sums / (variance * neighbor_count))
            
            # Realistic heatmaps should have positive spatial autocorrelation (Moran's I > 0.1)
            if moran_i < 0.05:
                self.failures.append(f"Data too uniform, Moran's I = {moran_i:.3f} (should be > 0.05)")
                return False
            
            # Check for extreme outliers (>3 std dev)
            outlier_ratio = np.sum(np.abs(data - data.mean()) > 3 * data.std()) / data.size
            if outlier_ratio > 0.05:
                self.failures.append(f"Too many outliers: {outlier_ratio:.2%}")
                return False
        
        except Exception as e:
            self.failures.append(f"Error in data structure validation: {e}")
            return False
        
        return True
    
    def validate_annotations(self, annotations, data_shape, fig_size_pixels):
        """Comprehensive annotation validation."""
        rows, cols = data_shape
        expected_cells = rows * cols
        
        # Count cells
        cell_count = sum(1 for ann in annotations if ann['class_id'] == 1)
        
        # Allow for some missing cells due to small size filtering, but expect most
        if cell_count < expected_cells * 0.90:  # At least 90% of cells should be detected
            self.failures.append(f"Missing too many cells: {cell_count}/{expected_cells}")
            return False
        
        # Check bbox validity
        for ann in annotations:
            bbox = ann['bbox']
            if not self._is_valid_bbox(bbox, fig_size_pixels):
                self.failures.append(f"Invalid bbox: {bbox}")
                return False
        
        # Check for duplicate bboxes (allow some tolerance for floating point)
        bboxes = [(ann['class_id'], ann['bbox']) for ann in annotations]
        unique_bboxes = set()
        for class_id, bbox in bboxes:
            # Round coordinates to avoid floating point precision issues
            rounded_bbox = tuple(round(coord, 1) for coord in [bbox.x0, bbox.y0, bbox.x1, bbox.y1])
            key = (class_id, rounded_bbox)
            if key in unique_bboxes:
                self.failures.append("Duplicate annotations detected")
                return False
            unique_bboxes.add(key)
        
        return True
    
    def validate_visual_elements(self, ax, annotations):
        """Check required elements are present."""
        class_ids = set(ann['class_id'] for ann in annotations)
        
        required = {0, 1}  # chart, cell - basic elements
        if not required.issubset(class_ids):
            self.failures.append(f"Missing required classes: {required - class_ids}")
            return False
        
        # Check colorbar presence (should exist in most heatmap cases)
        has_colorbar = 3 in class_ids
        if not has_colorbar and len([a for a in annotations if a['class_id'] == 0]) > 0:  # if there are charts
            # Only warn, don't fail - colorbars are not always present
            pass
        
        return True
    
    def _is_valid_bbox(self, bbox, fig_size):
        """Check bbox is within figure bounds and has positive area."""
        x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
        w, h = fig_size
        
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return False
        if x1 <= x0 or y1 <= y0:
            return False
        
        return True
    
    def generate_report(self):
        """Generate validation report."""
        if not self.failures:
            return "PASS: All quality checks passed"
        else:
            return "FAIL:\n" + "\n".join(f"- {f}" for f in self.failures)


def compute_spatial_autocorrelation(data):
    """Compute simplified Moran's I for spatial autocorrelation."""
    if data.size <= 1:
        return 0.0
    
    rows, cols = data.shape
    if rows < 2 or cols < 2:
        return 0.0
    
    mean_val = data.mean()
    variance = np.var(data)
    
    if variance == 0:
        return 0.0
    
    # Calculate neighbor relationships
    neighbor_sums = 0
    neighbor_count = 0
    
    for i in range(rows):
        for j in range(cols):
            current_val = data[i, j]
            
            # Check neighbors (up, down, left, right)
            neighbors = []
            if i > 0: neighbors.append(data[i-1, j])  # up
            if i < rows-1: neighbors.append(data[i+1, j])  # down
            if j > 0: neighbors.append(data[i, j-1])  # left
            if j < cols-1: neighbors.append(data[i, j+1])  # right
            
            for neighbor_val in neighbors:
                neighbor_sums += (current_val - mean_val) * (neighbor_val - mean_val)
                neighbor_count += 1
    
    if neighbor_count == 0:
        return 0.0
    
    moran_i = (neighbor_count / (rows * cols)) * (neighbor_sums / (variance * neighbor_count))
    return moran_i


def monitor_generation_quality(num_samples=100):
    """
    Generate validation set and compute quality metrics.
    This function can be used to monitor the heatmap generation quality.
    """
    print(f"Starting quality monitoring with {num_samples} samples...")
    
    metrics = {
        'data_autocorrelation': [],
        'annotation_completeness': [],
        'bbox_validity': [],
        'colormap_appropriateness': []
    }
    
    # Since we can't easily generate samples here, this is more for documentation
    # of how quality monitoring should work
    
    print("Quality monitoring completed. This function is available for validation checks.")
    return metrics

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--num', type=int, default=None)
    args = parser.parse_args()

    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        cfg = config_module.OCR_TRAINING_CONFIG
    else:
        cfg = GENERATION_CONFIG

    if args.num:
        cfg['num_images'] = args.num

    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    
    # Check if debug mode is enabled via environment variable
    if os.environ.get('DEBUG_MODE', '').lower() in ['true', '1', 'yes']:
        cfg['debug_mode'] = True
    
    if cfg['debug_mode']:
        print("--- DEBUG MODE ENABLED ---")
        debug_dir = 'test'
        print(f"Output will be saved to '{debug_dir}/'")
        images_dir = debug_dir
        labels_dir = debug_dir
        output_dir = debug_dir  # Define output_dir for debug mode
        ensure_dir(debug_dir)
    else:
        output_dir = cfg['output_dir']
        images_dir = os.path.join(output_dir, 'images')
        labels_dir = os.path.join(output_dir, 'labels')
        ensure_dir(images_dir)
        ensure_dir(labels_dir)

    chart_generators = {
        "bar": _generate_bar_chart,
        "line": _generate_line_chart,
        "scatter": _generate_scatter_chart,
        "box": _generate_boxplot_chart,
        "pie": _generate_pie_chart,
        "area": _generate_area_chart,
        "histogram": _generate_histogram,
        "heatmap": _generate_heatmap_chart,
    }

    if cfg['debug_mode']:
        print(f"DEBUG: Available chart types: {list(cfg['chart_types'].keys())}")
        print(f"DEBUG: Enabled chart types: {[k for k, v in cfg['chart_types'].items() if v['enabled']]}")
        print(f"DEBUG: Scenario weights: {cfg['scenario_weights']}")
        print(f"DEBUG: Number of images to generate: {cfg['num_images']}")

    start_time = time.time()
    for i in range(cfg['num_images']):
        iter_start = time.time()
        print(f"--- Generating image {i+1}/{cfg['num_images']} ---")
        plt.close('all')

        output_dpi = random.choice([96, 120, 150])
        
        if cfg['debug_mode']:
            print(f"DEBUG: Using DPI {output_dpi}")

        # Determine scenario
        scenarios, weights = zip(*cfg['scenario_weights'].items())
        scenario = random.choices(scenarios, weights=weights, k=1)[0]
        
        if cfg['debug_mode']:
            print(f"DEBUG: Selected scenario: {scenario}")
        
        if scenario == 'multi':
            nrows, ncols = random.choice([(1,2), (2,1), (2,2), (1,3), (3,1), (2,3), (3,2), (3,3)])
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4), dpi=output_dpi)
            axes = np.array(axes).flatten()
            
            if cfg['debug_mode']:
                print(f"DEBUG: Created multi-axis chart: {nrows}x{ncols}")
        else:
            fig, ax = plt.subplots(figsize=(7, 5), dpi=output_dpi)
            axes = [ax]
            
            if cfg['debug_mode']:
                print(f"DEBUG: Created single-axis chart: 1x1")

        chart_info_map = {}

        for ax_idx, ax in enumerate(axes):
            chart_types, weights = zip(*[(k, v['weight']) for k, v in cfg['chart_types'].items() if v['enabled']])
            chart_type = random.choices(chart_types, weights=weights, k=1)[0]
            print(f"  - Type: {chart_type} (Scenario: {scenario})")
            
            if cfg['debug_mode']:
                print(f"DEBUG: AX[{ax_idx}]: Chart type selected: {chart_type}")
                print(f"DEBUG: AX[{ax_idx}]: Chart types and weights: {list(zip(chart_types, weights))}")

            # Get correct class map for this chart type
            cls_map = CHART_CLASS_MAPS.get(chart_type, CHART_CLASS_MAPS['bar'])
            
            if cfg['debug_mode']:
                print(f"DEBUG: AX[{ax_idx}]: Class map: {cls_map}")
            
            theme_name = random.choice(list(THEMES.keys()))
            theme_config = THEMES[theme_name]
            print(f"  - Theme: {theme_name}")
            
            if cfg['debug_mode']:
                print(f"DEBUG: AX[{ax_idx}]: Theme selected: {theme_name}")

            is_scientific = random.random() < cfg['bar_chart_config']['scientific_ratio']
            style_config = {}

            generator_func = chart_generators[chart_type]
            
            # Initialize all possible return variables
            boxplot_dict = {}
            scale_axis_info = {}
            keypoint_data = None
            
            if chart_type == 'box':
                if cfg['debug_mode']:
                    print(f"DEBUG: AX[{ax_idx}]: Calling box plot generator")
                # FIX: _generate_boxplot_chart returns (data_artists, other_artists, bar_info_list, orientation,
                #      error_tops, axis_related_artists, scale_axis_info, boxplot_metadata)
                # Position 6 = scale_axis_info (contains boxplot_raw), Position 7 = boxplot_metadata
                data_artists, other_artists, bar_info_list, orientation, error_tops, axis_related_artists, scale_axis_info, boxplot_dict = generator_func(ax, theme_name, theme_config, is_scientific, debug_mode=cfg['debug_mode'])
                keypoint_data = None  # Box plots don't have keypoint data
            else:
                if cfg['debug_mode']:
                    print(f"DEBUG: AX[{ax_idx}]: Calling generator for {chart_type}")
                if chart_type == 'bar':
                    if cfg['debug_mode']:
                        print(f"DEBUG: AX[{ax_idx}]: Setting up bar chart style config")
                    styles, weights = zip(*[(k, v['weight']) for k, v in cfg['bar_chart_config']['styles'].items()])
                    style_config['style'] = random.choices(styles, weights=weights, k=1)[0]
                    patterns, weights = zip(*[(k, v['weight']) for k, v in cfg['bar_chart_config']['patterns'].items()])
                    style_config['pattern'] = random.choices(patterns, weights=weights, k=1)[0]
                    style_config['is_scientific'] = is_scientific
                
                # Handle the 8-element return from all chart generators
                result = generator_func(ax, theme_name, theme_config, style_config if chart_type == 'bar' else is_scientific, debug_mode=cfg['debug_mode'])
                
                # Check if the result has 8 elements (new enhanced functions) or 7 (legacy)
                if len(result) == 8:
                    data_artists, other_artists, bar_info_list, orientation, error_tops, axis_related_artists, scale_axis_info, keypoint_data = result
                else:  # Legacy function returning 7 elements
                    data_artists, other_artists, bar_info_list, orientation, error_tops, axis_related_artists, scale_axis_info = result
                    keypoint_data = None
            
            other_artists.extend(axis_related_artists)
            
            if cfg['debug_mode']:
                print(f"DEBUG: AX[{ax_idx}]: Generated {len(data_artists)} data artists, {len(other_artists)} other artists")
                print(f"DEBUG: AX[{ax_idx}]: Scale axis info: {scale_axis_info}")
            
            ax.set_title(random.choice(CHART_TITLES), fontsize=14, pad=15)
            if random.random() < 0.6 and chart_type not in ['pie', 'heatmap']:
                ax.legend(loc=random.choice(['upper right', 'best']))

            chart_info_map[ax] = {
                'chart_type_str': chart_type,
                'data_artists': data_artists,
                'other_artists': other_artists,
                'axis_related_artists': axis_related_artists,
                'boxplot_dict': boxplot_dict,
                'scale_axis_info': scale_axis_info,
                'keypoint_info': keypoint_data,  
                'pie_geometry': keypoint_data if chart_type == 'pie' else None  
            }

            # Store keypoint metadata for line, area, and pie charts
            if chart_type in ['line', 'area'] and keypoint_data is not None:
                chart_info_map[ax]['keypoint_info'] = keypoint_data
            elif chart_type == 'pie' and keypoint_data is not None:
                chart_info_map[ax]['pie_geometry'] = keypoint_data

        fig.tight_layout(pad=2.0)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=output_dpi)
        buf.seek(0)

        # Determine the primary chart type for annotation extraction
        primary_chart_type = chart_info_map.get(axes[0], {}).get('chart_type_str', 'bar')
        cls_map = CHART_CLASS_MAPS.get(primary_chart_type, CHART_CLASS_MAPS['bar'])
        
        if cfg['debug_mode']:
            print(f"DEBUG: Primary chart type for annotation: {primary_chart_type}")
            print(f"DEBUG: Using class map for annotations: {cls_map}")

        annotations = get_granular_annotations(fig, chart_info_map, cls_map)
        
        if cfg['debug_mode']:
            print(f"DEBUG: Total annotations detected: {len(annotations)}")
            class_counts = {}
            for ann in annotations:
                class_id = ann['class_id']
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
            print(f"DEBUG: Annotation class distribution: {class_counts}")

        pil_img_for_check = Image.open(buf).convert('RGB')
        img_w, img_h = pil_img_for_check.size

        # Filter low-variance annotations
        filtered_annotations = []
        PIXEL_STD_DEV_THRESHOLD = 10

        # Get class IDs for axis_labels and legend
        axis_labels_class_id = next((k for k, v in cls_map.items() if v == 'axis_labels'), None)
        legend_class_id = next((k for k, v in cls_map.items() if v == 'legend'), None)

        for ann in annotations:
            if (axis_labels_class_id is not None and ann['class_id'] == axis_labels_class_id) or \
               (legend_class_id is not None and ann['class_id'] == legend_class_id):
                bbox = ann['bbox']
                
                if legend_class_id is not None and ann['class_id'] == legend_class_id:
                    padding = 3
                    x0 = max(bbox.x0 + padding, bbox.x0)
                    y0 = max(bbox.y0 + padding, bbox.y0)
                    x1 = max(bbox.x1 - padding, x0 + 1)
                    y1 = max(bbox.y1 - padding, y0 + 1)
                    ann['bbox'] = BoundingBox(x0, y0, x1, y1)
                else:
                    x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
                
                crop_box = (int(x0), int(img_h - y1), int(x1), int(img_h - y0))

                if crop_box[0] < crop_box[2] and crop_box[1] < crop_box[3]:
                    label_crop = pil_img_for_check.crop(crop_box)
                    l_crop = label_crop.convert('L')
                    std_dev = np.array(l_crop).std()

                    if std_dev > PIXEL_STD_DEV_THRESHOLD:
                        filtered_annotations.append(ann)
            else:
                filtered_annotations.append(ann)
        
        annotations = filtered_annotations

        # Dual-axis post-processing
        if len(fig.axes) == 2 and any(v == 'axis_labels' for v in cls_map.values()):
            print("    - Applying post-processing fix for dual-axis chart annotations.")
            axis_label_class_id = next((k for k, v in cls_map.items() if v == 'axis_labels'), None)
            if axis_label_class_id is not None:
                renderer = fig.canvas.get_renderer()
                main_ax_bbox = axes[0].get_window_extent(renderer)
                xaxis_y_threshold = main_ax_bbox.y0
                
                if cfg['debug_mode']:
                    print(f"DEBUG: Dual-axis processing - threshold Y: {xaxis_y_threshold}")
                
                x_axis_labels_to_filter = []
                annotations_to_keep = []
                
                for ann in annotations:
                    is_class_8 = (ann['class_id'] == axis_label_class_id)
                    is_on_x_axis = (ann['bbox'].y1 < xaxis_y_threshold)
                    
                    if is_class_8 and is_on_x_axis:
                        x_axis_labels_to_filter.append(ann)
                    else:
                        annotations_to_keep.append(ann)
                
                deduplicated_x_labels = []
                seen_x_positions = []
                tolerance = 5
                
                for ann in sorted(x_axis_labels_to_filter, key=lambda a: a['bbox'].x0):
                    x_center = (ann['bbox'].x0 + ann['bbox'].x1) / 2
                    is_duplicate = any(abs(x_center - seen_x) < tolerance for seen_x in seen_x_positions)
                    
                    if not is_duplicate:
                        deduplicated_x_labels.append(ann)
                        seen_x_positions.append(x_center)
                
                annotations = annotations_to_keep + deduplicated_x_labels
                print(f"    - Kept {len(deduplicated_x_labels)} of {len(x_axis_labels_to_filter)} X-axis labels.")
                
                if cfg['debug_mode']:
                    print(f"DEBUG: After dual-axis processing - annotations: {len(annotations)}")

        # Apply realism effects
        buf.seek(0)
        pil_img = Image.open(buf).convert('RGB')
        
        if cfg['debug_mode']:
            print(f"DEBUG: Before realism effects - annotations: {len(annotations)}")
        
        pil_img, annotations = apply_realism_effects(pil_img, annotations, cfg['realism_effects'])
        
        if cfg['debug_mode']:
            print(f"DEBUG: After realism effects - annotations: {len(annotations)}")

        # Filter annotations by size and aspect ratio
        MIN_BBOX_SIZE = 8
        MAX_ASPECT_RATIO = 20.0
        
        valid_annotations = []
        for ann in annotations:
            bbox = ann['bbox']
            width = bbox.x1 - bbox.x0
            height = bbox.y1 - bbox.y0
                
            if chart_type == 'scatter' or chart_type == 'box':
                if width == 0 or height == 0:
                    continue
                valid_annotations.append(ann)
            else:     
                if width >= MIN_BBOX_SIZE and height >= MIN_BBOX_SIZE :
                    if width > 0 and height > 0:
                        aspect_ratio = max(width / height, height / width)
                        if aspect_ratio > MAX_ASPECT_RATIO:
                            print(f"    - Discarded annotation (extreme aspect ratio): {aspect_ratio:.1f}")
                            continue
                        valid_annotations.append(ann)
                else:
                    print(f"    - Discarded annotation (too small): class {ann['class_id']}, size {width:.1f}x{height:.1f}")

        annotations = valid_annotations
    
        # Filter out-of-bounds annotations
        img_w, img_h = pil_img.size
        final_valid_annotations = []
        for ann in annotations:
            bbox = ann['bbox']
            if (bbox.x0 >= 0 and bbox.y0 >= 0 and 
                bbox.x1 <= img_w and bbox.y1 <= img_h and
                bbox.x1 > bbox.x0 and bbox.y1 > bbox.y0):
                final_valid_annotations.append(ann)
            else:
                print(f"    - Discarded annotation (out of bounds after effects): class {ann['class_id']}")

        annotations = final_valid_annotations
        
        if cfg['debug_mode']:
            print(f"DEBUG: After size/aspect/bounds filtering - annotations: {len(annotations)}")
        
        annotations = filter_overlapping_annotations(annotations, iou_threshold=0.7)
        
        if cfg['debug_mode']:
            print(f"DEBUG: After overlap filtering - annotations: {len(annotations)}")

        # Save files
        base_filename = f"chart_{i:05d}"
        
        pil_img.save(os.path.join(images_dir, f"{base_filename}.png"))
        
        # Save YOLO format labels
        save_annotations_yolo(annotations, img_w, img_h, 
                             os.path.join(labels_dir, f"{base_filename}.txt"))


        # **NEW: Check if chart type is area and save dual-format annotations**
        primary_chart_type = chart_info_map.get(fig.axes[0], {}).get('chart_type_str', '')

        if primary_chart_type == 'area':
            # Save object detection format with CLASS_MAP_AREA_OBJ
            clsmap_obj = GENERATION_CONFIG['CLASS_MAP_AREA_OBJ']
            annotations_obj = get_granular_annotations(fig, chart_info_map, clsmap_obj)
            
            # Create separate directory for area object annotations
            area_obj_dir = os.path.join(output_dir, 'area_obj_labels')
            ensure_dir(area_obj_dir)
            save_annotations_yolo(annotations_obj, img_w, img_h, 
                                os.path.join(area_obj_dir, f"{base_filename}.txt"))
            
            # Save pose format with CLASS_MAP_AREA_POSE
            clsmap_pose = GENERATION_CONFIG['CLASS_MAP_AREA_POSE']
            # Convert string keys to reverse map
            clsmap_pose_reverse = {v: k for k, v in clsmap_pose.items()}
            
            keypoint_annotations = extract_area_pose_annotations_fixed(
                fig, chart_info_map, clsmap_pose_reverse, img_w, img_h
            )
            
            # Create separate directory for area pose annotations
            area_pose_dir = os.path.join(output_dir, 'area_pose_labels')
            ensure_dir(area_pose_dir)
            save_annotations_pose_fixed(keypoint_annotations, img_w, img_h,
                                os.path.join(area_pose_dir, f"{base_filename}.txt"))
            
            if cfg['debug_mode']:
                print(f"DEBUG: Saved {len(annotations_obj)} area object annotations")
                print(f"DEBUG: Saved {len(keypoint_annotations)} area pose annotations")

        elif primary_chart_type == 'pie':
            # **NEW: Save pie chart dual-format annotations**
            # Save object detection format with CLASS_MAP_PIE_OBJ
            clsmap_obj = GENERATION_CONFIG['CLASS_MAP_PIE_OBJ']
            annotations_obj = get_granular_annotations(fig, chart_info_map, clsmap_obj)
            
            pie_obj_dir = os.path.join(output_dir, 'pie_obj_labels')
            ensure_dir(pie_obj_dir)
            save_annotations_yolo(annotations_obj, img_w, img_h,
                                os.path.join(pie_obj_dir, f"{base_filename}.txt"))
            
            # Save pose format with CLASS_MAP_PIE_POSE
            clsmap_pose = GENERATION_CONFIG['CLASS_MAP_PIE_POSE']
            clsmap_pose_reverse = {v: k for k, v in clsmap_pose.items()}
            
            keypoint_annotations = extract_pie_pose_annotations_fixed(
                fig, chart_info_map, clsmap_pose_reverse, img_w, img_h
            )
            
            pie_pose_dir = os.path.join(output_dir, 'pie_pose_labels')
            ensure_dir(pie_pose_dir)
            save_annotations_pose_fixed(keypoint_annotations, img_w, img_h,
                                os.path.join(pie_pose_dir, f"{base_filename}.txt"))
            
            if cfg['debug_mode']:
                print(f"DEBUG: Saved {len(annotations_obj)} pie object annotations")
                print(f"DEBUG: Saved {len(keypoint_annotations)} pie pose annotations")

        elif primary_chart_type == 'line':
            # **NEW: Save line chart dual-format annotations**
            # Save object detection format with CLASS_MAP_LINE_OBJ
            clsmap_obj = GENERATION_CONFIG['CLASS_MAP_LINE_OBJ']
            annotations_obj = get_granular_annotations(fig, chart_info_map, clsmap_obj)
            
            line_obj_dir = os.path.join(output_dir, 'line_obj_labels')
            ensure_dir(line_obj_dir)
            save_annotations_yolo(annotations_obj, img_w, img_h,
                                os.path.join(line_obj_dir, f"{base_filename}.txt"))
            
            # Save pose format with CLASS_MAP_LINE_POSE
            clsmap_pose = GENERATION_CONFIG['CLASS_MAP_LINE_POSE']
            clsmap_pose_reverse = {v: k for k, v in clsmap_pose.items()}
            
            keypoint_annotations = extract_line_pose_annotations_fixed(
                fig, chart_info_map, clsmap_pose_reverse, img_w, img_h
            )
            
            line_pose_dir = os.path.join(output_dir, 'line_pose_labels')
            ensure_dir(line_pose_dir)
            save_annotations_pose_fixed(keypoint_annotations, img_w, img_h,
                                os.path.join(line_pose_dir, f"{base_filename}.txt"))
            
            if cfg['debug_mode']:
                print(f"DEBUG: Saved {len(annotations_obj)} line object annotations")
                print(f"DEBUG: Saved {len(keypoint_annotations)} line pose annotations")
        
        # NEW: Handle 5 specific chart types - save to dedicated directories
        elif primary_chart_type in ['bar', 'histogram', 'scatter', 'box', 'heatmap']:
            # Use the appropriate class map for each chart type
            cls_map_specific = CHART_CLASS_MAPS.get(primary_chart_type, CHART_CLASS_MAPS['bar'])
            annotations_obj = get_granular_annotations(fig, chart_info_map, cls_map_specific)
            
            # Create directory name based on chart type
            if primary_chart_type == 'box':
                # Use 'box' for directory name instead of 'box'
                obj_dir = os.path.join(output_dir, 'box_obj_labels')
            else:
                obj_dir = os.path.join(output_dir, f"{primary_chart_type}_obj_labels")
            
            ensure_dir(obj_dir)
            save_annotations_yolo(annotations_obj, img_w, img_h,
                                os.path.join(obj_dir, f"{base_filename}.txt"))
            
            if cfg['debug_mode']:
                print(f"DEBUG: Saved {len(annotations_obj)} {primary_chart_type} object annotations to {obj_dir}")
        
        # Determine the appropriate class map for unified JSON based on chart type
        cls_map = CHART_CLASS_MAPS.get(primary_chart_type, CHART_CLASS_MAPS['bar'])

        # Create three separate JSON files as expected by merge_json.py

        # Get comprehensive unified JSON with complete metadata
        unified_json = create_unified_annotation(fig, chart_info_map, cls_map, img_w, img_h, annotations)
        unified_json = convert_numpy_types(unified_json)

        # 1. Create detailed JSON with element annotations and all metadata
        detailed_json = {
            "chart_type": unified_json.get("chart_type"),
            "orientation": unified_json.get("orientation"),
            "scale_labels": unified_json.get("scale_labels", []),
            "tick_labels": unified_json.get("tick_labels", []),
            "chart_title": unified_json.get("chart_title", []),
            "axis_title": unified_json.get("axis_title", []),
            "legend": unified_json.get("legend", []),
            "bar": unified_json.get("bar", []),
            "datapoint": unified_json.get("datapoint", []),
            "errorbar": unified_json.get("errorbar", []),
            "significancemarker": unified_json.get("significancemarker", []),
            "datalabel": unified_json.get("datalabel", []),
            "box": unified_json.get("box", []),
            "medianline": unified_json.get("medianline", []),
            "rangeindicator": unified_json.get("rangeindicator", []),
            "outlier": unified_json.get("outlier", []),
            "wedge": unified_json.get("wedge", []),
            "linesegment": unified_json.get("linesegment", []),
            "areaboundary": unified_json.get("areaboundary", []),
            "cell": unified_json.get("cell", []),
            "colorbar": unified_json.get("colorbar", []),
            "connectorline": unified_json.get("connectorline", []),

            # Include all the new metadata in the detailed file
            "scale_axis_info": unified_json.get("scale_axis_info", {}),
            "bar_info": unified_json.get("bar_info", []),
            "keypoint_info": unified_json.get("keypoint_info", []),
            "boxplot_metadata": unified_json.get("boxplot_metadata", {}),
            "pie_geometry": unified_json.get("pie_geometry", {}),
            "series_count": unified_json.get("series_count", 1),
            "series_names": unified_json.get("series_names", []),
            "stacking_mode": unified_json.get("stacking_mode"),
            "dual_axis_info": unified_json.get("dual_axis_info", {}),
            "style": unified_json.get("style"),
            "pattern": unified_json.get("pattern"),
            "is_scientific": unified_json.get("is_scientific", False),
            "raw_annotations": unified_json.get("raw_annotations", [])
        }

        # 2. Create OCR JSON with OCR annotations
        # For now, create an empty structure that will be populated by OCR processing
        ocr_json = {
            "ocr_annotations": [],  # This would normally come from OCR processing
            "effects_applied": []   # This might be added during image processing
        }

        # 3. Create basic metadata JSON
        metadata_json = {
            "image_id": base_filename,
            "resolution": [int(img_w), int(img_h)],
            "chart_types": [unified_json.get("chart_type", "unknown")],
            "themes": {},  # Will be populated based on the chart theme
            "num_annotations": len(annotations)
        }

        # Save the three files
        with open(os.path.join(labels_dir, f"{base_filename}_detailed.json"), 'w') as f:
            json.dump(detailed_json, f, indent=2)

        with open(os.path.join(labels_dir, f"{base_filename}_ocr.json"), 'w') as f:
            json.dump(ocr_json, f, indent=2)

        with open(os.path.join(labels_dir, f"{base_filename}.json"), 'w') as f:
            json.dump(metadata_json, f, indent=2)

        iter_time = time.time() - iter_start
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (cfg['num_images'] - i - 1)
        print(f"    ✓ Complete in {iter_time:.2f}s | ETA: {eta/60:.1f}m")
        print(f" ✓ Saved {len(annotations)} annotations")

    print("\n=== DATASET STATISTICS ===")
    class_counts = defaultdict(int)
    for i in range(cfg['num_images']):
        label_file = os.path.join(labels_dir, f"chart_{i:05d}.txt")
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1

    # Use the combined class map for statistics
    combined_cls_map = {}
    for chart_type, chart_cls_map in CHART_CLASS_MAPS.items():
        if chart_type != 'pie':  # Skip pie since it has empty mapping
            for id_val, class_name in chart_cls_map.items():
                if id_val not in combined_cls_map:
                    combined_cls_map[id_val] = class_name

    for class_id, class_name in sorted(combined_cls_map.items(), key=lambda x: x[1]):
        print(f"  {class_name:20s}: {class_counts[class_id]:5d} instances")

    # Merge JSON files if enabled in config
    if cfg.get('merge_json_files', False) and batch_merge_all:
        print("\n--- Merging JSON files ---")
        batch_merge_all(labels_dir)
    
    if cfg['debug_mode']:
        print("\n--- Generation complete. Running visualization script... ---")
        try:
            subprocess.run(["python", "testar.py", debug_dir, "--show"], check=True)
        except FileNotFoundError:
            print("\n[ERROR] Could not find 'python'. Make sure Python is in your system's PATH.")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] 'testar.py' script failed with error: {e}")
        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred while trying to run testar.py: {e}")


if __name__ == '__main__':
    main()