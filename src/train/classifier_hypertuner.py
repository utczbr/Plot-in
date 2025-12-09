import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class HypertuningResult:
    chart_type: str
    optimal_params: Dict
    best_accuracy: float
    best_f1_score: float
    best_precision: float
    best_recall: float
    evaluation_metrics: Dict
    optimization_history: List[Dict]
    n_trials: int
    best_trial_number: int

class EnhancedChartClassifierHypertuner:
    """
    Advanced hyperparameter optimizer using Bayesian optimization (Optuna)
    with chart-type-specific parameter spaces and multi-objective optimization.
    """
    
    def __init__(self, chart_type: str, n_jobs: int = -1, verbose: bool = True):
        self.chart_type = chart_type
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.param_space = self._get_param_space_continuous(chart_type)
        self.param_space_discrete = self._get_param_space_discrete(chart_type)
        self.best_params = None
        self.study = None
        
        logger.info(f"Initialized hypertuner for {chart_type} charts")
    
    def _get_param_space_continuous(self, chart_type: str) -> Dict:
        """
        Define continuous parameter search space for Bayesian optimization
        Ranges are informed by empirical analysis and first_step.md recommendations
        """
        
        if chart_type == 'bar':
            return {
                # Position weights
                'scale_left_weight': (4.0, 8.0),
                'scale_right_weight': (3.5, 7.0),
                'scale_bottom_weight': (4.0, 7.5),
                'tick_bottom_weight': (5.0, 9.0),
                'tick_left_weight': (5.0, 8.5),
                
                # Context weights
                'bar_alignment_weight': (3.5, 7.0),
                'bar_spacing_weight': (3.0, 6.0),
                'numeric_boost': (2.0, 5.0),
                
                # Size thresholds
                'scale_size_max_width': (0.05, 0.12),
                'scale_size_max_height': (0.025, 0.06),
                
                # Aspect ratio
                'aspect_ratio_min': (0.3, 0.6),
                'aspect_ratio_max': (3.0, 5.0),
                'title_aspect_min': (4.0, 8.0),
                
                # Decision thresholds
                'classification_threshold': (1.0, 3.5),
                'confidence_margin_factor': (0.2, 0.6)
            }
        
        elif chart_type == 'line':
            return {
                # Position weights
                'left_edge_weight': (5.0, 8.5),
                'right_edge_weight': (4.0, 7.0),
                'bottom_edge_weight': (4.5, 8.0),
                'top_edge_weight': (2.0, 5.0),
                
                # Line-specific
                'line_proximity_weight': (3.5, 6.5),
                'value_range_weight': (2.5, 5.0),
                'numeric_boost': (3.0, 6.0),
                
                # Size thresholds
                'scale_size_max_width': (0.04, 0.10),
                'scale_size_max_height': (0.02, 0.05),
                
                # Title detection
                'title_size_min': (0.10, 0.18),
                'title_aspect_min': (5.0, 9.0),
                
                # Thresholds
                'classification_threshold': (1.5, 4.0),
                'edge_threshold_x': (0.15, 0.25),
                'edge_threshold_y': (0.75, 0.90)
            }
        
        elif chart_type == 'scatter':
            return {
                # Position weights
                'left_edge_weight': (5.5, 9.0),
                'right_edge_weight': (5.0, 8.0),
                'bottom_edge_weight': (5.0, 8.5),
                
                # Scatter-specific
                'point_cloud_proximity_weight': (4.0, 7.0),
                'dual_axis_penalty': (0.4, 1.2),
                'numeric_boost': (4.0, 7.0),
                
                # Size thresholds
                'scale_size_max_width': (0.04, 0.11),
                'scale_size_max_height': (0.025, 0.055),
                
                # Title detection
                'title_size_min': (0.11, 0.20),
                'title_aspect_min': (5.5, 10.0),
                
                # Thresholds
                'classification_threshold': (2.0, 4.5),
                'edge_threshold': (0.12, 0.25)
            }
        
        elif chart_type == 'box':
            return {
                # Position weights
                'scale_edge_weight': (4.5, 8.0),
                'tick_alignment_weight': (4.0, 7.5),
                
                # Box-specific
                'box_spacing_weight': (3.0, 6.0),
                'category_weight': (2.5, 5.0),
                'numeric_boost': (2.5, 5.5),
                
                # Size thresholds
                'scale_size_max_width': (0.045, 0.095),
                'scale_size_max_height': (0.025, 0.055),
                
                # Title detection
                'title_size_min': (0.10, 0.18),
                'title_aspect_min': (5.0, 9.0),
                
                # Thresholds
                'classification_threshold': (1.5, 3.5),
                'edge_threshold': (0.15, 0.28)
            }
        
        elif chart_type == 'histogram':
            return {
                # Position weights
                'left_edge_weight': (5.0, 8.5),
                'bottom_edge_weight': (4.5, 8.0),
                'frequency_axis_weight': (4.5, 7.5),
                
                # Histogram-specific
                'bin_alignment_weight': (4.0, 7.0),
                'continuous_scale_weight': (2.5, 5.5),
                'numeric_boost': (3.0, 6.0),
                
                # Size thresholds
                'scale_size_max_width': (0.045, 0.10),
                'scale_size_max_height': (0.025, 0.055),
                
                # Title detection
                'title_size_min': (0.10, 0.18),
                'title_aspect_min': (5.0, 8.5),
                
                # Thresholds
                'classification_threshold': (1.5, 3.5),
                'edge_threshold': (0.13, 0.25)
            }
        
        else:
            return {}
    
    def _get_param_space_discrete(self, chart_type: str) -> Dict:
        """Define discrete parameter values for grid search"""
        continuous = self._get_param_space_continuous(chart_type)
        
        discrete = {}
        for param_name, (min_val, max_val) in continuous.items():
            # Generate 5-7 discrete values spanning the range
            n_steps = 6
            discrete[param_name] = np.linspace(min_val, max_val, n_steps).tolist()
        
        return discrete
    
    def _create_optuna_study(self, 
                            sampler_type: str = 'tpe',
                            pruner_type: str = 'median',
                            n_startup_trials: int = 20) -> optuna.Study:
        """
        Create Optuna study with specified sampler and pruner
        
        Args:
            sampler_type: 'tpe' (Tree-structured Parzen Estimator) or 'cmaes'
            pruner_type: 'median' or 'hyperband'
            n_startup_trials: Number of random trials before Bayesian optimization
        """
        
        # Select sampler
        if sampler_type == 'tpe':
            sampler = TPESampler(
                n_startup_trials=n_startup_trials,
                n_ei_candidates=24,
                seed=42
            )
        elif sampler_type == 'cmaes':
            sampler = CmaEsSampler(seed=42)
        else:
            sampler = TPESampler(seed=42)
        
        # Select pruner
        if pruner_type == 'median':
            pruner = MedianPruner(
                n_startup_trials=n_startup_trials,
                n_warmup_steps=10
            )
        elif pruner_type == 'hyperband':
            pruner = HyperbandPruner(
                min_resource=10,
                max_resource=100,
                reduction_factor=3
            )
        else:
            pruner = MedianPruner()
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=f"{self.chart_type}_optimization"
        )
        
        return study
    
    def optimize_bayesian(self,
                         training_data: List[Dict],
                         validation_data: List[Dict],
                         n_trials: int = 100,
                         timeout: Optional[int] = None,
                         sampler_type: str = 'tpe',
                         pruner_type: str = 'median',
                         use_cv: bool = False,
                         n_folds: int = 5) -> HypertuningResult:
        """
        Bayesian optimization using Optuna with TPE sampler
        
        Args:
            training_data: Training dataset
            validation_data: Validation dataset
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds
            sampler_type: 'tpe' or 'cmaes'
            pruner_type: 'median' or 'hyperband'
            use_cv: Use cross-validation instead of fixed validation split
            n_folds: Number of CV folds if use_cv=True
        """
        
        logger.info(f"Starting Bayesian optimization for {self.chart_type}")
        logger.info(f"Sampler: {sampler_type}, Pruner: {pruner_type}")
        logger.info(f"Trials: {n_trials}, Timeout: {timeout}s")
        
        # Create Optuna study
        self.study = self._create_optuna_study(sampler_type, pruner_type)
        
        # Define objective function
        def objective(trial: optuna.Trial) -> float:
            # Sample parameters from search space
            params = {}
            for param_name, (min_val, max_val) in self.param_space.items():
                params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            
            # Evaluate parameters
            if use_cv:
                # Cross-validation evaluation
                cv_scores = []
                all_data = training_data + validation_data
                
                # Create stratified folds if possible
                try:
                    y_labels = [sample.get('chart_type_id', 0) for sample in all_data]
                    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                    folds = list(skf.split(all_data, y_labels))
                except:
                    # Fallback to simple splits
                    fold_size = len(all_data) // n_folds
                    folds = [(list(range(i*fold_size, (i+1)*fold_size)), 
                             list(range((i+1)*fold_size, len(all_data))) + list(range(0, i*fold_size)))
                            for i in range(n_folds)]
                
                for fold_idx, (train_idx, val_idx) in enumerate(folds):
                    fold_train = [all_data[i] for i in train_idx]
                    fold_val = [all_data[i] for i in val_idx]
                    
                    metrics = self._evaluate_params(params, fold_val)
                    cv_scores.append(metrics['f1_score'])
                    
                    # Report intermediate value for pruning
                    trial.report(np.mean(cv_scores), fold_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                return np.mean(cv_scores)
            else:
                # Standard train/validation split
                train_metrics = self._evaluate_params(params, training_data)
                val_metrics = self._evaluate_params(params, validation_data)
                
                # Combined score: prioritize validation F1, with train F1 as regularization
                combined_score = (
                    0.75 * val_metrics['f1_score'] + 
                    0.15 * val_metrics['precision'] +
                    0.10 * train_metrics['f1_score']
                )
                
                # Report intermediate value
                trial.report(combined_score, 0)
                
                return combined_score
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=self.verbose
        )
        
        # Extract best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        # Evaluate best parameters on validation set
        final_metrics = self._evaluate_params(best_params, validation_data)
        
        # Extract optimization history
        history = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params
                })
        
        logger.info(f"Optimization complete. Best F1: {best_value:.4f}")
        logger.info(f"Best trial: #{self.study.best_trial.number}")
        
        result = HypertuningResult(
            chart_type=self.chart_type,
            optimal_params=best_params,
            best_accuracy=final_metrics['accuracy'],
            best_f1_score=final_metrics['f1_score'],
            best_precision=final_metrics['precision'],
            best_recall=final_metrics['recall'],
            evaluation_metrics=final_metrics,
            optimization_history=history,
            n_trials=len(self.study.trials),
            best_trial_number=self.study.best_trial.number
        )
        
        self.best_params = best_params
        return result
    
    def _evaluate_params(self, params: Dict, data: List[Dict]) -> Dict:
        """
        Evaluate parameter configuration on dataset with robust metrics
        """
        if not data:
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0
            }
        
        # Import classifiers dynamically
        try:
            if self.chart_type == 'bar':
                from ..core.classifiers.bar_chart_classifier import BarChartClassifier
                classifier = BarChartClassifier(params)
            elif self.chart_type == 'line':
                from ..core.classifiers.line_chart_classifier import LineChartClassifier
                classifier = LineChartClassifier(params)
            elif self.chart_type == 'scatter':
                from ..core.classifiers.scatter_chart_classifier import ScatterChartClassifier
                classifier = ScatterChartClassifier(params)
            elif self.chart_type == 'box':
                from ..core.classifiers.box_chart_classifier import BoxChartClassifier
                classifier = BoxChartClassifier(params)
            elif self.chart_type == 'histogram':
                from ..core.classifiers.histogram_chart_classifier import HistogramChartClassifier
                classifier = HistogramChartClassifier(params)
            else:
                raise ValueError(f"Unknown chart type: {self.chart_type}")
        except ImportError as e:
            logger.error(f"Failed to import classifier: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                   'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        
        # Evaluate on data
        tp, fp, fn, tn = 0, 0, 0, 0
        
        for sample in data:
            try:
                result = classifier.classify(
                    axis_labels=sample['axis_labels'],
                    chart_elements=sample.get('chart_elements', []),
                    img_width=sample['img_width'],
                    img_height=sample['img_height'],
                    orientation=sample.get('orientation', 'vertical')
                )
                
                # Compare with ground truth using IoU matching
                gt_scale = set([self._bbox_to_key(lbl['xyxy']) for lbl in sample.get('gt_scale_labels', [])])
                gt_tick = set([self._bbox_to_key(lbl['xyxy']) for lbl in sample.get('gt_tick_labels', [])])
                gt_title = set([self._bbox_to_key(lbl['xyxy']) for lbl in sample.get('gt_axis_titles', [])])
                
                pred_scale = set([self._bbox_to_key(lbl['xyxy']) for lbl in result.scale_labels])
                pred_tick = set([self._bbox_to_key(lbl['xyxy']) for lbl in result.tick_labels])
                pred_title = set([self._bbox_to_key(lbl['xyxy']) for lbl in result.axis_titles])
                
                # Compute TP, FP, FN
                tp += len(gt_scale & pred_scale) + len(gt_tick & pred_tick) + len(gt_title & pred_title)
                fp += len(pred_scale - gt_scale) + len(pred_tick - gt_tick) + len(pred_title - gt_title)
                fn += len(gt_scale - pred_scale) + len(gt_tick - pred_tick) + len(gt_title - pred_title)
                
            except Exception as e:
                logger.warning(f"Error evaluating sample: {e}")
                continue
        
        # Compute metrics with safe division
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    @staticmethod
    def _bbox_to_key(bbox: List[float]) -> str:
        """Convert bbox to hashable string key"""
        return f"{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}"
    
    def save_results(self, result: HypertuningResult, output_path: str):
        """Save hypertuning results to JSON"""
        output_data = asdict(result)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history using matplotlib"""
        try:
            import matplotlib.pyplot as plt
            
            if self.study is None:
                logger.warning("No study available to plot")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot optimization history
            trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            values = [t.value for t in trials]
            trial_numbers = [t.number for t in trials]
            
            ax1.plot(trial_numbers, values, 'b-', alpha=0.6, label='Trial value')
            ax1.axhline(y=self.study.best_value, color='r', linestyle='--', label=f'Best: {self.study.best_value:.4f}')
            ax1.set_xlabel('Trial Number')
            ax1.set_ylabel('Objective Value (F1 Score)')
            ax1.set_title(f'{self.chart_type.capitalize()} - Optimization History')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot parameter importance (if available)
            try:
                importance = optuna.importance.get_param_importances(self.study)
                params = list(importance.keys())[:10]  # Top 10
                importances = [importance[p] for p in params]
                
                ax2.barh(params, importances)
                ax2.set_xlabel('Importance')
                ax2.set_title('Parameter Importance (Top 10)')
                ax2.grid(True, alpha=0.3, axis='x')
            except:
                ax2.text(0.5, 0.5, 'Parameter importance\nnot available',
                        ha='center', va='center', transform=ax2.transAxes)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Chart Classifier Hyperparameter Optimization')
    parser.add_argument('--chart-type', required=True, 
                       choices=['bar', 'line', 'scatter', 'box', 'histogram', 'all'],
                       help='Chart type to optimize')
    parser.add_argument('--data-dir', required=True, help='Directory containing training data')
    parser.add_argument('--method', default='bayesian',
                       choices=['bayesian', 'grid', 'random'],
                       help='Optimization method')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=None, help='Optimization timeout in seconds')
    parser.add_argument('--sampler', default='tpe', choices=['tpe', 'cmaes'], 
                       help='Bayesian sampler type')
    parser.add_argument('--pruner', default='median', choices=['median', 'hyperband'],
                       help='Trial pruner type')
    parser.add_argument('--cv', action='store_true', help='Use cross-validation')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs')
    parser.add_argument('--output', required=True, help='Output JSON path')
    parser.add_argument('--plot', action='store_true', help='Generate optimization plots')
    
    args = parser.parse_args()
    
    # TODO: Load training and validation data from args.data_dir
    # training_data, validation_data = load_data(args.data_dir)
    
    logger.info(f"Starting optimization for {args.chart_type} using {args.method} method")
    
    # For demonstration - replace with actual data loading
    training_data = []
    validation_data = []
    
    if args.chart_type == 'all':
        chart_types = ['bar', 'line', 'scatter', 'box', 'histogram']
    else:
        chart_types = [args.chart_type]
    
    all_results = {}
    
    for chart_type in chart_types:
        logger.info(f"\n{'='*60}\nOptimizing {chart_type} classifier\n{'='*60}")
        
        tuner = EnhancedChartClassifierHypertuner(
            chart_type=chart_type,
            n_jobs=args.n_jobs,
            verbose=True
        )
        
        if args.method == 'bayesian':
            result = tuner.optimize_bayesian(
                training_data=training_data,
                validation_data=validation_data,
                n_trials=args.n_trials,
                timeout=args.timeout,
                sampler_type=args.sampler,
                pruner_type=args.pruner,
                use_cv=args.cv,
                n_folds=args.n_folds
            )
        else:
            logger.error(f"Method {args.method} not yet implemented")
            continue
        
        all_results[chart_type] = asdict(result)
        
        # Save individual result
        output_path = Path(args.output).parent / f"{chart_type}_hypertuning_result.json"
        tuner.save_results(result, str(output_path))
        
        # Plot if requested
        if args.plot:
            plot_path = Path(args.output).parent / f"{chart_type}_optimization_history.png"
            tuner.plot_optimization_history(str(plot_path))
    
    # Save combined results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nAll results saved to {args.output}")
