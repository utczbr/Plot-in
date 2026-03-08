# Training Strategies for Scoring Functions

In the context of scoring functions like the ones in the chart analysis code (e.g., `_compute_octant_region_scores` and `_compute_multi_feature_scores`), which rely on heuristics such as Gaussian kernels, thresholds, and weighted features for classifying elements (e.g., scale labels, tick labels), "training" can refer to two main paradigms: (1) optimizing heuristic parameters through empirical methods, or (2) evolving them into machine learning (ML) models for more adaptive scoring. The former treats scores as rule-based with tunable hyperparameters (e.g., sigmas, weights like `+= 5.0`), while the latter uses supervised learning to predict scores or classes based on features (e.g., centroid, aspect ratio, text content).

Based on established practices in chart parsing, graph analysis, and general ML scoring functions (e.g., in computer vision or molecular docking), here are key strategies. These draw from techniques like synthetic data generation for chart understanding, debiasing datasets, diagnostic tools like learning curves, and nonlinear model training. I'll outline general approaches, then specifics for type-specific functions as you suggested.

### 1. **Heuristic Optimization Strategies**
If keeping the functions rule-based (non-ML), focus on tuning parameters for better accuracy without full retraining. This is lightweight and suitable for targeted per-graph-type refinement.

- **Empirical Tuning and Grid Search**: Manually adjust parameters (e.g., `sigma_left = 0.008`, thresholds like `nx < 0.20`) on a validation set of labeled charts. Use grid search or random search to explore combinations, evaluating on metrics like precision/recall for element classification. For each graph type (e.g., bar vs. line), create separate configs to target training—e.g., wider sigmas for scatter plots with dispersed elements.
  
- **Ablation Studies**: Systematically remove/add features (e.g., disable `numeric_ratio` in scoring) and measure impact on a benchmark dataset. This helps isolate effective rules per type, similar to how modularization in chart synthesis identifies key components.

- **Dataset-Driven Calibration**: Collect or synthesize charts with ground-truth labels (e.g., via tools like ChartParser or manual annotation). Optimize weights to minimize loss (e.g., mean squared error between predicted and true scores) using optimization libraries like SciPy's `minimize`.

| Strategy | Pros | Cons | Best for Graph Types |
|----------|------|------|----------------------|
| Grid Search | Exhaustive, finds optimal params | Computationally expensive for many params | Simple types (e.g., bar, pie) with few variables |
| Ablation | Interpretable, low overhead | May miss interactions between features | All types; start with vertical/horizontal orientations |
| Calibration | Data-adaptive without ML overhead | Requires labeled data | Complex types (e.g., scatter, line) needing fine tweaks |

### 2. **Machine Learning-Based Training Strategies**
Convert heuristics to ML by treating features (e.g., normalized_pos, aspect_ratio, region_scores) as inputs to a model predicting scores (regression) or classes (classification, e.g., softmax over 'scale_label', etc.). This enables targeted training per graph type via separate models or multi-task learning. Use frameworks like PyTorch or scikit-learn.

- **Supervised Learning Setup**:
  - **Data Preparation**: Use labeled datasets like ChartQA, Chart2Text, or synthetic ones. Generate synthetic charts via pipelines that modularize generation (e.g., separate data tables from plotting functions using Matplotlib and GPT-like tools), then diversify visuals (e.g., add annotations, vary fonts) to mimic real charts. This improves element classification by 10-20% on benchmarks like CharXiv, as it bridges synthetic-real gaps.
  - **Debiasing**: Filter data to avoid leakage—e.g., remove charts with >80% similarity in layout or content between train/test sets (using metrics like Fréchet Inception Distance for images or Tanimoto for features). This prevents memorization, ensuring generalizable scoring, with performance drops (e.g., correlation from 0.80 to 0.75) indicating successful bias removal.
  - **Model Architectures**: Start with simple nonlinear models like Random Forest (RF) or Support Vector Regression (SVR) for scoring, which capture complex interactions better than linear heuristics (e.g., R_p up to 0.80 vs. 0.64 for rules). For spatial features, use Graph Neural Networks (GNNs) like E(n)-equivariant ones, where nodes are chart elements and edges encode distances/alignments—pretrain on pose-like tasks (e.g., element alignment) then finetune on scores.

- **Training Protocols**:
  - **Two-Stage Training**: Pretrain on abundant proxy tasks (e.g., classifying regions via synthetic data), then finetune on specific scores using MSE or cross-entropy loss. Use LoRA (low-rank adaptation) for efficient tuning on per-type subsets, with 1 epoch on ~10k samples yielding 10-15% gains.
  - **Modular and Type-Specific Training**: Train separate models per graph type (e.g., bar-focused on spacing alignment, line on numeric trends) or use multi-head architectures. Diversify training data across 20-30 chart types and themes for robustness, with filtering (e.g., retain high-confidence QA pairs) to optimize quality.
  - **Optimization Techniques**: Apply early stopping based on validation loss. Use learning rates like 1e-4, batch sizes of 32-96. For family-specific (graph-type) models, employ leave-cluster-out cross-validation to tailor without overfitting.

- **Diagnostic and Improvement Tools**:
  - **Learning Curves**: Plot training/validation loss over epochs to diagnose issues—flat training loss signals underfit (add complexity); diverging validation loss indicates overfit (add regularization). For scoring tasks, monitor metrics like MSE; unrepresentative data shows large gaps (collect more samples).
  - **Feature Selection**: Use data-driven methods (e.g., RF feature importance) to prioritize inputs like `region_scores` or `numeric_ratio`, reducing noise. Include augmentation like docking decoys for VS-like scoring.

|             ML Strategy        |          Key Techniques             |         Performance Impact            |          Relevance to Chart Scoring                |
|--------------------------------|-------------------------------------|---------------------------------------|----------------------------------------------------|
| Data Synthesis & Modularization | Separate data/plot generation, conditional subplots | +10-20% accuracy on element tasks | Enhances targeted training for diverse charts (e.g., multi-subplot) |
| Debiasing & Filtering | Similarity thresholds, confidence scoring | Reduces inflated metrics by 5-10% | Prevents bias in type-specific models |
| Nonlinear Models (RF/SVR/GNN) | Ensemble learning, equivariance | R_p/Spearman up to 0.80 | Better than heuristics for complex features like orientations |
| Two-Stage Finetuning | Pretrain on proxies, finetune on scores | +5-15% on benchmarks | Efficient for limited labeled chart data |

### Implementation Tips for Your Code
- **Targeted Per-Type Training**: Refactor into a dispatcher calling type-specific models (e.g., `score_bar_chart` as an RF trained on bar datasets). Start with synthetic data generation: Use Matplotlib to create 10k+ charts, label via OCR/tools, then train.
- **Evaluation**: Use benchmarks like CASF analogs for charts (e.g., ChartBench). Metrics: Precision/recall for classification, correlation for scores.
- **Scalability**: For large datasets, use GPU-accelerated training (e.g., 4 A100s for 1 epoch). If resources are limited, begin with heuristic tuning before ML.
