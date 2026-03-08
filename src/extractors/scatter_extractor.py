"""
Extractor for scatter plots with mode-specific processing.

§3b.2: Supports 2D Gaussian sub-pixel refinement (feature flag: scatter_subpixel_mode).
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
import cv2

from extractors.base_extractor import BaseExtractor
from utils.geometry_utils import find_closest_element

class ScatterExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()

    def extract(self, img, detections, scale_model, baseline_coord, img_dimensions, mode='optimized', x_scale_model=None, y_baseline_coord=None, x_baseline_coord=None):
        data_points = detections.get('data_point', [])
        data_labels = detections.get('data_label', [])
        error_bars = detections.get('error_bar', [])

        # Use BaseExtractor to create result template
        result = self._create_result_template('scatter', detections, len(data_points))

        r_squared = img_dimensions.get('r_squared', None)
        advanced_settings = img_dimensions.get('advanced_settings', {})
        subpixel_mode = advanced_settings.get('scatter_subpixel_mode', 'otsu')

        # Use provided baseline coordinates or extract from single baseline_coord
        if y_baseline_coord is None or x_baseline_coord is None:
            if baseline_coord is not None:
                y_baseline_coord = baseline_coord
                x_baseline_coord = None
            else:
                y_baseline_coord = img_dimensions.get('y_baseline_coord', None)
                x_baseline_coord = img_dimensions.get('x_baseline_coord', None)

        # Resolve scale functions using BaseExtractor helper
        y_scale_func = self._resolve_scale_func(scale_model)
        x_scale_func = self._resolve_scale_func(x_scale_model)

        # Pre-compute grayscale image for Gaussian mode (avoids per-point conversion)
        gray_img = None
        if subpixel_mode == 'gaussian' and len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif subpixel_mode == 'gaussian':
            gray_img = img

        for i, point in enumerate(data_points):
            x1, y1, x2, y2 = point['xyxy']
            x_center_int, y_center_int = (x1 + x2) / 2, (y1 + y2) / 2

            box_width = int(x2 - x1)
            box_height = int(y2 - y1)
            x_center, y_center = x_center_int, y_center_int

            if box_width > 2 and box_height > 2:
                if subpixel_mode == 'gaussian' and gray_img is not None:
                    # §3b.2: 2D Gaussian sub-pixel refinement
                    result_gf = self._refine_subpixel_gaussian(
                        gray_img, (x1, y1, x2, y2), pad=2
                    )
                    if result_gf is not None:
                        x_center, y_center = result_gf[0], result_gf[1]
                else:
                    # Legacy: Otsu-based refinement
                    try:
                        pad = 2
                        x1_p = max(0, int(x1) - pad)
                        y1_p = max(0, int(y1) - pad)
                        x2_p = min(img.shape[1], int(x2) + pad)
                        y2_p = min(img.shape[0], int(y2) + pad)
                        crop = img[y1_p:y2_p, x1_p:x2_p]
                        if crop.size > 0:
                            if len(crop.shape) == 3:
                                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            else:
                                gray_crop = crop
                            if np.mean(gray_crop) > 127:
                                gray_crop = 255 - gray_crop
                            _, thresh = cv2.threshold(
                                gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                            )
                            M = cv2.moments(thresh)
                            if M["m00"] > 0:
                                x_center = x1_p + M["m10"] / M["m00"]
                                y_center = y1_p + M["m01"] / M["m00"]
                    except Exception:
                        pass

            y_calibrated = float(y_scale_func(y_center)) if y_scale_func else None
            x_calibrated = float(x_scale_func(x_center)) if x_scale_func else None

            point_info = {
                'index': i,
                'xyxy': point['xyxy'],
                'confidence': point.get('conf', 0.0),
                'x_pixel': x_center,
                'y_pixel': y_center,
                'x_calibrated': x_calibrated,
                'y_calibrated': y_calibrated,
                'x_baseline_distance': x_center - x_baseline_coord if x_baseline_coord is not None else None,
                'y_baseline_distance': y_center - y_baseline_coord if y_baseline_coord is not None else None,
                'data_label': None,
                'error_bar': None
            }

            assoc_data_label = find_closest_element(point, data_labels, orientation='vertical')
            if assoc_data_label:
                point_info['data_label'] = {
                    'text': assoc_data_label.get('text', ''),
                    'value': assoc_data_label.get('cleanedvalue'),
                    'bbox': assoc_data_label.get('xyxy')
                }

            assoc_error_bar = find_closest_element(point, error_bars, orientation='vertical')
            if assoc_error_bar:
                eb_x1, eb_y1, eb_x2, eb_y2 = assoc_error_bar['xyxy']
                error_margin = {}
                if scale_model:
                    try:
                        error_margin['y_margin'] = abs(float(y_scale_func(eb_y1)) - float(y_scale_func(eb_y2)))
                    except Exception: pass
                if x_scale_model:
                    try:
                        error_margin['x_margin'] = abs(float(x_scale_func(eb_x1)) - float(x_scale_func(eb_x2)))
                    except Exception: pass
                point_info['error_bar'] = {'margins': error_margin, 'bbox': assoc_error_bar['xyxy']}

            result['data_points'].append(point_info)
        
        if result['data_points']:
            x_vals = [p['x_calibrated'] for p in result['data_points'] if p['x_calibrated'] is not None]
            y_vals = [p['y_calibrated'] for p in result['data_points'] if p['y_calibrated'] is not None]

            result['statistics'] = {
                'x_mean': float(np.mean(x_vals)) if x_vals else None,
                'y_mean': float(np.mean(y_vals)) if y_vals else None,
                'x_std': float(np.std(x_vals)) if x_vals else None,
                'y_std': float(np.std(y_vals)) if y_vals else None,
                'count': len(result['data_points'])
            }

            if len(x_vals) > 1 and len(y_vals) > 1 and len(x_vals) == len(y_vals) and np.std(x_vals) > 0 and np.std(y_vals) > 0:
                try:
                    correlation_matrix = np.corrcoef(x_vals, y_vals)
                    result['correlation'] = float(correlation_matrix[0, 1])
                except Exception:
                    result['correlation'] = None
        
        # Store metadata
        result['y_baseline_coord'] = y_baseline_coord
        result['x_baseline_coord'] = x_baseline_coord
        result['baseline_note'] = 'For scatter plots, baselines are calibration zeros (reference only), not used in point value calculations'
        
        result['calibration'] = {
            'x_axis': {
                'has_calibration': x_scale_model is not None,
                'x_baseline_coord': x_baseline_coord,
                'x_zero_crossing': x_baseline_coord
            },
            'y_axis': {
                'has_calibration': scale_model is not None,
                'y_baseline_coord': y_baseline_coord,
                'y_zero_crossing': y_baseline_coord
            }
        }
        
        # Use BaseExtractor helper for calibration quality
        if r_squared is not None:
             result['calibration_quality'] = {'r_squared': r_squared}

        return result

    # ── §3b.2: 2D Gaussian Sub-Pixel Refinement ─────────────────────────

    @staticmethod
    def _refine_subpixel_gaussian(
        gray_img: np.ndarray, bbox: Tuple, pad: int = 2
    ) -> Optional[Tuple[float, float, bool]]:
        """
        Fit an axis-aligned 2D Gaussian to the marker intensity patch and
        return the sub-pixel center (mu_x, mu_y, converged).

        §3b.2.1: G(x,y;θ) = A·exp(-(x-μx)²/(2σx²) - (y-μy)²/(2σy²)) + C
        §3b.2.2: Solved via scipy.optimize.least_squares (LM) with analytical Jacobian.
        §3b.2.3: Initialized from intensity moments.

        Staff Refinements:
        - Vectorized residuals (no Python loops over patch).
        - Analytical Jacobian for 500+ marker latency budget.
        - Robustness guard: reject if σ ∉ [0.3, 3.0] or non-convergent.

        Returns (mu_x_global, mu_y_global, converged) or None on failure.
        """
        try:
            from scipy.optimize import least_squares
        except ImportError:
            return None

        x1, y1, x2, y2 = bbox
        x1_p = max(0, int(x1) - pad)
        y1_p = max(0, int(y1) - pad)
        x2_p = min(gray_img.shape[1], int(x2) + pad)
        y2_p = min(gray_img.shape[0], int(y2) + pad)

        patch = gray_img[y1_p:y2_p, x1_p:x2_p].astype(float)
        if patch.size < 9:  # Need at least 3×3
            return None

        ph, pw = patch.shape

        # Pre-compute meshgrid (vectorized)
        yy, xx = np.mgrid[0:ph, 0:pw]
        xx_flat = xx.ravel().astype(float)
        yy_flat = yy.ravel().astype(float)
        I_flat = patch.ravel()

        # §3b.2.3: Initialization from intensity moments
        I_sum = np.sum(I_flat) + 1e-10
        C0 = float(np.min(I_flat))
        A0 = float(np.max(I_flat)) - C0
        if A0 < 1.0:
            return None  # Flat patch, no marker signal

        # Weighted centroid (using intensity above background)
        I_shifted = np.maximum(I_flat - C0, 0.0)
        I_shifted_sum = np.sum(I_shifted) + 1e-10
        mu_x0 = float(np.sum(I_shifted * xx_flat) / I_shifted_sum)
        mu_y0 = float(np.sum(I_shifted * yy_flat) / I_shifted_sum)
        sig_x0 = float(np.sqrt(np.sum(I_shifted * (xx_flat - mu_x0) ** 2) / I_shifted_sum))
        sig_y0 = float(np.sqrt(np.sum(I_shifted * (yy_flat - mu_y0) ** 2) / I_shifted_sum))
        sig_x0 = max(sig_x0, 0.5)
        sig_y0 = max(sig_y0, 0.5)

        # θ = [A, mu_x, mu_y, sigma_x, sigma_y, C]
        theta0 = np.array([A0, mu_x0, mu_y0, sig_x0, sig_y0, C0])

        def residuals(theta):
            A, mx, my, sx, sy, C = theta
            sx2 = max(sx * sx, 1e-8)
            sy2 = max(sy * sy, 1e-8)
            G = A * np.exp(-0.5 * ((xx_flat - mx) ** 2 / sx2 + (yy_flat - my) ** 2 / sy2)) + C
            return I_flat - G

        def jacobian(theta):
            A, mx, my, sx, sy, C = theta
            sx2 = max(sx * sx, 1e-8)
            sy2 = max(sy * sy, 1e-8)
            dx = xx_flat - mx
            dy = yy_flat - my
            exp_term = np.exp(-0.5 * (dx ** 2 / sx2 + dy ** 2 / sy2))
            G_noC = A * exp_term

            n = len(I_flat)
            J = np.empty((n, 6))
            J[:, 0] = -exp_term                        # ∂r/∂A
            J[:, 1] = -G_noC * dx / sx2                # ∂r/∂μx
            J[:, 2] = -G_noC * dy / sy2                # ∂r/∂μy
            J[:, 3] = -G_noC * dx ** 2 / (sx2 * sx)    # ∂r/∂σx
            J[:, 4] = -G_noC * dy ** 2 / (sy2 * sy)    # ∂r/∂σy
            J[:, 5] = -np.ones(n)                       # ∂r/∂C
            return J

        try:
            result = least_squares(
                residuals, theta0, jac=jacobian, method='lm',
                max_nfev=20, ftol=1e-6, xtol=1e-6
            )
        except Exception:
            return None

        A_fit, mx_fit, my_fit, sx_fit, sy_fit, C_fit = result.x
        converged = result.success

        # Robustness guard: reject implausible sigmas
        if not (0.3 <= abs(sx_fit) <= 3.0 * pw and 0.3 <= abs(sy_fit) <= 3.0 * ph):
            # Fall back to intensity centroid
            return (float(x1_p + mu_x0), float(y1_p + mu_y0), False)

        if not converged:
            return (float(x1_p + mu_x0), float(y1_p + mu_y0), False)

        # Map back to global image coordinates
        return (float(x1_p + mx_fit), float(y1_p + my_fit), True)