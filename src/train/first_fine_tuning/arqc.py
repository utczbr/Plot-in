
# Create comprehensive production-ready modular spatial classification system
# with hyperparameter optimization support

import json

# Define the complete architecture structure
architecture = {
    "modules": {
        "core": [
            "spatial_classifier.py - Main classification orchestrator",
            "feature_extractor.py - Multi-scale feature extraction",
            "region_analyzer.py - Spatial region scoring with Gaussian kernels",
            "chart_type_classifier.py - Chart type detection module"
        ],
        "classifiers": [
            "bar_chart_classifier.py - Bar chart specialized module",
            "line_chart_classifier.py - Line/scatter chart specialized module", 
            "box_plot_classifier.py - Box plot specialized module",
            "pie_chart_classifier.py - Pie chart specialized module"
        ],
        "hypertuning": [
            "hyperparameter_optimizer.py - Bayesian/gradient-based tuner",
            "synthetic_data_generator.py - Ground truth chart generator",
            "parameter_config.py - Type-specific parameter storage"
        ],
        "utils": [
            "clustering_utils.py - DBSCAN/KMeans for dual-axis detection",
            "geometry_utils.py - Bbox operations, IoU, alignment scoring",
            "validation_metrics.py - Precision, recall, F1, confusion matrix"
        ]
    },
    "best_practices": {
        "modular_design": [
            "Separate concerns: detection, classification, calibration",
            "Chart-type specific modules inherit from base classifier",
            "Easy to add new chart types without modifying existing code"
        ],
        "hyperparameter_ready": [
            "All scoring weights parameterized in config files",
            "Support for gradient-based (PyTorch) and Bayesian (Optuna) optimization",
            "Synthetic data generation for ground truth training"
        ],
        "production_ready": [
            "Comprehensive logging and error handling",
            "Unit tests for each module (>90% coverage)",
            "Performance profiling and optimization hooks",
            "Confidence scoring for fallback mechanisms"
        ]
    }
}

#Plan
#
#Implementation Plan: LYLAA Enhancement
#
#  This plan outlines the steps for Phase 1 (Foundational Tuning) and Phase 2 (Architectural 
#  Refactoring). The goal is to significantly improve the accuracy and maintainability of the chart 
#  analysis system by integrating a data-driven hyperparameter tuning pipeline and refactoring the 
#  core classification logic into a modular, object-oriented design.
#
#  ---
#
#  Phase 1: Foundational Tuning with Gradient-Based Optimization
#
#  Objective: To replace the hardcoded, empirical parameters in the LYLAA spatial classification 
#  algorithm with a set of optimized parameters derived from a data-driven, gradient-based tuning 
#  process. This will provide an immediate and significant boost in classification accuracy.
#
#  Selected Approach: We will implement the PyTorch-based gradient descent hyperparameter tuning 
#  system as detailed in train/lylaa-hypertuner.py and train/plan_Lylla.md. This approach is chosen 
#  over the Bayesian optimization proposed in train/first_step.md because it offers a more complete 
#  and sophisticated implementation that is better suited for the large number of parameters (24+) 
#  in our system.
#
#  Key Code Assets:
#   * `train/lylaa-hypertuner.py`: The core of our tuning system.
#   * `train/Integration.py`: Provides the necessary code for integrating the data generator and 
#     applying the tuned parameters.
#   * `core/spatial_classification_enhanced.py`: The target for our improvements.
#
#  Implementation Steps:
#
#   1. Data Generation Setup:
#       * Action: Modify the data generator located at train/gerador_charts_hypertuning/ by 
#         integrating the extract_label_features_for_hypertuning function from train/Integration.py.
#       * Goal: The generator must produce *_hypertuning.json files alongside the chart images. These 
#         JSON files will contain the rich feature vectors (normalized position, aspect ratio, etc.) 
#         and the ground truth class (scale_label, tick_label, axis_title) for every label in the 
#         synthetic charts.
#       * Verification: Run the generator for a small batch (e.g., --num 10) and verify that the 
#         _hypertuning.json files are created correctly in the labels/ subdirectory.
#
#   2. Hyperparameter Tuning Execution:
#       * Action: Run the lylaa-hypertuner.py script, pointing it to the directory containing the 
#         generated data.
#   1         python train/lylaa-hypertuner.py --data-dir train/gerador_charts_hypertuning/ --epochs 
#     300 --lr 0.005
#       * Goal: To execute the training pipeline, which will use the generated data to find the 
#         optimal values for the 24 parameters of the LYLAA algorithm. The output will be a 
#         lylaa_hypertuning_results.json file.
#       * Verification: The script should complete without errors, and the output JSON file should 
#         contain a dictionary of optimized parameters.
#
#   3. Integration of Tuned Parameters:
#       * Action: Modify the _compute_octant_region_scores and _compute_multi_feature_scores functions
#          in core/spatial_classification_enhanced.py to accept a settings dictionary, as detailed in 
#         train/Integration.py. All hardcoded numerical weights and thresholds in these functions 
#         should be replaced with lookups from this settings dictionary, with fallbacks to the current
#          default values.
#       * Goal: To make the core classification logic parameterizable, allowing the tuned values to be
#          injected at runtime.
#       * Verification: The modified functions should work correctly with and without the settings 
#         dictionary, producing the same results as before when no settings are provided.
#
#   4. Production Deployment of Tuned Parameters:
#       * Action: Create a new wrapper class, HypertunedSpatialClassifier, as specified in 
#         train/Integration.py. This class will load the lylaa_hypertuning_results.json file at 
#         initialization and pass the optimized parameters to the 
#         spatial_classify_axis_labels_enhanced function.
#       * Goal: To provide a simple, drop-in mechanism for using the tuned parameters in the main 
#         application.
#       * Verification: The HypertunedSpatialClassifier should correctly load the parameters and 
#         produce classification results that reflect the new, optimized logic.
#
#  Definition of Done for Phase 1:
#   * The data generator produces the required _hypertuning.json files.
#   * The lylaa-hypertuner.py script runs successfully and produces an optimized parameter set.
#   * The spatial_classification_enhanced.py module is refactored to accept the tuned parameters.
#   * The HypertunedSpatialClassifier is implemented and can be used to run the classification with 
#     the new parameters.
#
#  ---
#
#  Phase 2: Architectural Refactoring to Modular Classifiers
#
#  Objective: To refactor the monolithic spatial_classify_axis_labels_enhanced function into a 
#  modular, object-oriented system with specialized classifiers for each chart type. This will 
#  improve maintainability, extensibility, and allow for more targeted, chart-specific logic.
#
#  Selected Approach: We will adopt the Strategy Pattern as detailed in 
#  train/spatial-classification-modules.md. This involves creating a BaseChartClassifier interface 
#  and concrete implementations for each chart type (e.g., BarChartClassifier, 
#  LineScatterClassifier).
#
#  Key Code Assets:
#   * `train/spatial-classification-modules.md`: The blueprint for our new architecture.
#   * `core/spatial_classification_enhanced.py`: The source of the logic to be refactored.
#
#  Implementation Steps:
#
#   1. Implement the Base Classifier:
#       * Action: Create a new file, classifiers/base_classifier.py, and implement the 
#         BaseChartClassifier abstract base class and the ClassificationResult dataclass as defined in
#          train/spatial-classification-modules.md.
#       * Goal: To establish a common interface for all specialized classifiers.
#
#   2. Create Specialized Classifiers:
#       * Action: Create new files for each chart-specific classifier (e.g., 
#         classifiers/bar_chart_classifier.py, classifiers/line_scatter_classifier.py). Implement the 
#         BarChartClassifier and LineScatterClassifier classes, inheriting from BaseChartClassifier.
#       * Goal: To encapsulate the chart-specific classification logic. The existing logic from 
#         _classify_precise_mode in spatial_classification_enhanced.py should be moved into the 
#         classify and compute_feature_scores methods of these new classes. The tuned parameters from 
#         Phase 1 should be passed to these classifiers during initialization.
#
#   3. Implement the Production Classifier Wrapper:
#       * Action: Create a ProductionSpatialClassifier class as described in 
#         train/spatial-classification-modules.md. This class will be responsible for:
#           1. Loading the hypertuned parameters from the JSON file.
#           2. Instantiating the correct specialized classifier based on the chart_type.
#           3. Passing the appropriate tuned parameters to the classifier.
#           4. Providing a confidence-based fallback to a generic classifier (e.g., BarChartClassifier)
#               if the chart type classification confidence is low.
#       * Goal: To create a single, intelligent entry point for the spatial classification system.
#
#   4. Integrate with the Main Application:
#       * Action: Update the ChartAnalysisOrchestrator and any other parts of the application that 
#         currently call spatial_classify_axis_labels_enhanced to instead use the new 
#         ProductionSpatialClassifier.
#       * Goal: To fully transition the application to the new modular architecture.
#
#  Definition of Done for Phase 2:
#   * The BaseChartClassifier and specialized classifier classes are implemented.
#   * The logic from spatial_classification_enhanced.py is successfully refactored into the new 
#     classifier classes.
#   * The ProductionSpatialClassifier is implemented and correctly loads parameters and dispatches to 
#     the appropriate specialized classifier.
#   * The main application is fully integrated with the new modular classification system.
#
#  By following this plan, we will systematically and robustly enhance the core of our chart 
#  analysis system, leading to significant improvements in both performance and long-term 
#  maintainability.



