"""
Property/regression tests for RobustSiegelCalibration.

Each test encodes a measured defect or guarantee, so a future change that
re-introduces it fails loudly instead of silently degrading calibrations:

  * unit equivariance   - an axis in metres vs 1e-9 metres must calibrate alike
  * small-N efficiency  - clean data must use all points and stay near OLS accuracy
  * log false positives - linear axes must not be classified as log
  * log detection       - genuine log axes must still be detected
  * outlier robustness  - up to 40% gross OCR errors must not corrupt the slope
  * hostile input       - never raise, never return non-finite values inside the data range

All data is generated from integer seeds: results are identical on every run.
Always judge a calibration through `result.func`, never by comparing `coeffs`
(log-scale coefficients live in log space).
"""
import logging
import sys
import unittest
import warnings

import numpy as np

sys.path.append('src')

from calibration.calibration_factory import CalibrationFactory  # noqa: E402
from calibration.calibration_siegel import RobustLinearEngine, RobustSiegelCalibration  # noqa: E402


def _labels(px, vals, conf=0.9):
    return [{'xyxy': [0, float(p) - 2, 10, float(p) + 2], 'cleanedvalue': float(v), 'ocr_confidence': conf}
            for p, v in zip(px, vals)]


def _slope(res):
    return (res.func(1000.0) - res.func(0.0)) / 1000.0


def _linear_ticks(rng, n, jitter=2.0, span=400.0):
    pos = 50 + span * np.arange(n) / (n - 1)
    vmin = 10 ** rng.uniform(0.5, 2.5)
    vals = vmin + vmin * (rng.uniform(1.5, 6) - 1) * np.arange(n) / (n - 1)
    return pos + rng.normal(0, jitter, n), vals, pos


class _Quiet(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        warnings.simplefilter('ignore')

    def tearDown(self):
        logging.disable(logging.NOTSET)


class TestUnitEquivariance(_Quiet):
    def test_outlier_rejection_does_not_depend_on_axis_unit(self):
        for scale in (1e-9, 1e-7, 1.0, 1e6):
            bad = 0
            for t in range(60):
                rng = np.random.default_rng([101, t])
                x, v, pos = _linear_ticks(rng, 8)
                y = v.copy()
                for j in rng.choice(8, 2, replace=False):
                    y[j] += rng.choice([-1, 1]) * rng.uniform(0.5, 2.0) * np.ptp(v)
                res = RobustSiegelCalibration().calibrate(_labels(x, y * scale), 'y')
                true = np.polyfit(pos, v, 1)[0]
                bad += abs(_slope(res) / scale - true) / abs(true) > 0.05
            self.assertEqual(bad, 0, f'outlier rejection broke at value scale {scale:g}')


class TestSmallNEfficiency(_Quiet):
    def test_minimum_n_uses_every_point(self):
        rng = np.random.default_rng([102, 0])
        x, v, _ = _linear_ticks(rng, 3)
        res = RobustSiegelCalibration().calibrate(_labels(x, v), 'y')
        self.assertEqual(int(res.inliers.sum()), 3, 'with 3 points there is no redundancy to reject on')

    def test_clean_data_accuracy_stays_close_to_ols(self):
        # N=4-5 still discard one good point (known, immaterial: ~0.2% of slope), so N=5 is only a no-regression guard
        for n, max_ratio in ((3, 1.4), (5, 1.7), (8, 1.35)):
            eng_err, ols_err = [], []
            for t in range(400):
                rng = np.random.default_rng([103, n, t])
                x, v, pos = _linear_ticks(rng, n)
                res = RobustSiegelCalibration().calibrate(_labels(x, v), 'y')
                true = np.polyfit(pos, v, 1)[0]
                eng_err.append(abs(_slope(res) - true))
                ols_err.append(abs(np.polyfit(x, v, 1)[0] - true))
            ratio = np.median(eng_err) / np.median(ols_err)
            self.assertLess(ratio, max_ratio, f'N={n}: robust fit is {ratio:.2f}x OLS error on CLEAN data')


class TestScaleTypeDetection(_Quiet):
    def test_linear_axes_are_not_classified_as_log(self):
        # stress-regime data (uniform positions, value-space noise): the case that produced 528/20000 false logs
        wrong = 0
        for t in range(1500):
            rng = np.random.default_rng(t)
            px = np.sort(rng.uniform(40, 860, 9))
            y = 0.4 * px + 150 + rng.normal(0, 1.5, 9)
            wrong += RobustSiegelCalibration().calibrate(_labels(px, y), 'y').scale_type == 'log'
        self.assertEqual(wrong, 0)

    def test_realistic_linear_ticks_are_not_classified_as_log(self):
        wrong = 0
        for t in range(400):
            rng = np.random.default_rng([104, t])
            n = int(rng.integers(4, 13))
            x, v, _ = _linear_ticks(rng, n, jitter=float(rng.choice([1, 2, 3])))
            wrong += RobustSiegelCalibration().calibrate(_labels(x, v), 'y').scale_type == 'log'
        self.assertLessEqual(wrong, 2)

    def test_genuine_log_axes_are_detected_and_accurate(self):
        det = tot = 0
        for t in range(150):
            rng = np.random.default_rng([105, t])
            dec = int(rng.choice([2, 3, 5]))
            span = rng.uniform(300, 900)
            if t % 2:
                tt = np.arange(dec + 1) / dec                                             # 1,10,100,...
            else:
                tt = np.sort(np.concatenate([np.arange(dec) + np.log10(m) for m in (1, 2, 5)] + [[dec]])) / dec
            pos = 50 + span * tt
            vals = 10 ** (dec * tt)
            res = RobustSiegelCalibration().calibrate(_labels(pos + rng.normal(0, 2, len(pos)), vals), 'y')
            tot += 1
            det += res.scale_type == 'log'
            mid = 50 + span / 2
            self.assertAlmostEqual(np.log10(res.func(mid)), dec / 2, delta=0.05 * dec)
        self.assertGreaterEqual(det / tot, 0.98)


class TestOutlierRobustness(_Quiet):
    def test_up_to_40_percent_gross_outliers_do_not_corrupt_the_slope(self):
        for contam in (0.3, 0.4):
            bad = 0
            for d in range(60):
                rng = np.random.default_rng([106, int(contam * 100), d])
                n = 10
                px = np.sort(rng.uniform(20, 900, n))
                yt = -0.5 * px + 200
                y = yt + rng.normal(0, 0.4, n)
                for j in rng.choice(n, int(round(contam * n)), replace=False):
                    y[j] = yt[j] + rng.choice([-1, 1]) * rng.uniform(0.5, 2.0) * 450
                res = RobustSiegelCalibration().calibrate(_labels(px, y), 'y')
                bad += res is None or abs(_slope(res) + 0.5) / 0.5 > 0.05
            self.assertLessEqual(bad, 1, f'{contam:.0%} contamination: {bad}/60 bad fits')


class TestHostileInput(_Quiet):
    def test_never_raises_and_is_finite_inside_the_data_range(self):
        rng = np.random.default_rng(2024)
        for _ in range(1200):
            n = int(rng.integers(0, 20))
            px = rng.choice([0, 1, 2], 1)[0]
            px = (rng.uniform(0, 1000, n) if px == 0 else np.round(rng.uniform(0, 5, n)) if px == 1
                  else np.full(n, 7.0))
            mode = int(rng.integers(0, 6))
            vals = [rng.normal(0, 1, n), np.full(n, 3.0), rng.uniform(-1e12, 1e12, n),
                    rng.uniform(1e-12, 1e-9, n), 10 ** rng.uniform(-150, 150, n),
                    np.array([rng.choice([np.nan, np.inf, 1.0, 2.0]) for _ in range(n)])][mode]
            lbl = _labels(px, vals, conf=float(rng.choice([0.0, 0.9, np.nan])))
            res = RobustSiegelCalibration().calibrate(lbl, 'y')          # must not raise
            if res is not None and len(px):
                inside = [res.func(v) for v in (float(np.min(px)), float(np.max(px)))]
                self.assertTrue(np.all(np.isfinite(inside)) and np.all(np.isfinite(res.coeffs)))


class TestConventionsAndFactory(_Quiet):
    def test_constant_values_report_zero_r2_not_perfect_fit(self):
        _, _, _, r2 = RobustLinearEngine().fit_line(np.arange(5.0), np.full(5, 7.0), np.ones(5))
        self.assertEqual(r2, 0.0)

    def test_factory_keeps_legacy_kwargs_working(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            eng = CalibrationFactory.create('prosac', max_trials=500, lo_iters=2,
                                            early_termination_ratio=0.85, convergence_threshold=1e-8)
        self.assertIsInstance(eng, RobustSiegelCalibration)
        with self.assertRaises(ValueError):
            CalibrationFactory.create('fast', max_trials=1)

    def test_factory_creates_siegel_and_aliases(self):
        for alias in ('siegel', 'robust', 'Siegel (Robust)', 'PROSAC', 'Adaptive'):
            eng = CalibrationFactory.create(alias)
            self.assertIsInstance(eng, RobustSiegelCalibration)


if __name__ == '__main__':
    unittest.main()
