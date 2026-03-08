# Study Protocol

## Evaluating PlotToData, a Tool for Automated Data Extraction from Scientific Plots

**Tamires Martins¹, Gabriel da Silva Stuart¹, Vitória Soares dos Santos¹, Alexandra Bannach-Brown²**

¹Universidade Federal de Santa Catarina (UFSC)
²Berlin Institute of Health (BIH), Charité – Universitätsmedizin Berlin

---

## Background

The increasing accumulation of scientific publications necessitates effective methods for synthesizing information, such as systematic reviews and meta-analyses (Ioannidis, 2023). Conducting systematic reviews is time-consuming, particularly in fields with a high volume of publications (Bannach-Brown et al., 2021; Bugajska et al., 2025). Among the data collected, quantitative data extraction is one of the most time-consuming and error-prone processes, which may influence the results of the meta-analysis (Cramond et al., 2019; Van der Mierden et al., 2021). The most common methods involve using digital rulers and digital applications, such as the measuring tool in Adobe Acrobat and WebPlotDigitizer (Cramond et al., 2019). Rulers are less precise methodologies, and although available applications improve accuracy and reduce time requirements, they still leave the data collection process entirely dependent on the reviewer (Cramond et al., 2019; Kadic et al., 2016).

A new tool, PlotToData, has been developed to extract data from multiple plots within diverse PDFs simultaneously in a fully automated manner. The system is developed in Python and extracts structured data from plots using computer vision, optical character recognition (OCR), and mathematical algorithms. The tool imports user-selected PDFs or image files, detects and classifies plot images, and extracts their data through a 10-stage processing pipeline. Users can filter, review, and correct the extracted values directly in the graphical interface, and the final cleaned dataset can be exported as a CSV file. The tool is available as open-source code on GitHub; ONNX detection and classification models are currently distributed via the installer manifest (HuggingFace hosting is planned for a future release). Validation of this tool is necessary to identify potential performance limitations, provide users with guidance on its use, and assess performance levels across diverse plot types.

---

## Aims

i. Evaluate the performance of PlotToData in extracting data from different types of plots within PDF files.
ii. Collect user feedback on the tool's usability and intuitiveness.

---

## Tool Name

**PlotToData**
*(candidate names evaluated during development: Plot-in Data-out, Plot-out, Plot-in)*

---

## Version

1.0.0

---

## Testers

To be determined. Testers should include at least two independent reviewers with experience in systematic review data extraction, to enable inter-rater reliability assessment.

---

## Tool and Its Functionalities

PlotToData supports data extraction from scientific plots of eight types: **bar, line, scatter, box, histogram, heatmap, area, and pie**. The architecture is organized by coordinate system (Cartesian, Grid, and Polar) as shown in Figure 1.

### Entry Points

- **Command-line interface (CLI):** `src/analysis.py`
- **Graphical user interface (GUI):** `src/main_modern.py`
- **Programmatic / batch service:** `src/core/analysis_manager.py`

### Processing Pipeline

The tool executes a 10-stage pipeline per image asset:

1. **Input resolution and PDF extraction (T1):** The user selects a single file or a folder containing PDF files, image files, or a mixed combination (`--input-type auto | image | pdf`). PDFs are rasterized via PyMuPDF/OpenCV into individual chart images. Each asset carries provenance metadata (`source_document`, `page_index`, `figure_id`).

2. **Chart classification (T2/T3):** A trained ONNX classification model (`classification.onnx`) assigns each image to one of the eight canonical chart types. When only a generic class is returned, the system defaults to `bar`. Users may override model settings through the interface or CLI flags.

3. **Element detection and model routing (T3/T4):** A chart-type-specific ONNX detection model is loaded. Bar, line, scatter, box, histogram, heatmap, and area plots use bounding-box output; pie plots use pose-keypoint output. Histogram detection includes a two-tier fallback chain: reduced-confidence retry, then remapping from the bar model.

4. **Orientation detection (T3/T4):** For bar, histogram, and box plots, an `OrientationDetectionService` determines horizontal vs. vertical orientation using variance, aspect ratio, and spatial distribution. Non-Cartesian chart types default to vertical.

5. **OCR and text-region merge (T3/T5):** Axis labels and optional DocLayout text regions are merged and deduplicated (IoU-based), enriching detections with text and confidence values for downstream context matching.

6. **Handler dispatch (T4):** A `ChartAnalysisOrchestrator` routes each image to the appropriate handler: `CartesianExtractionHandler` (bar, line, scatter, box, histogram, area), `GridChartHandler` (heatmap), or `PolarChartHandler` (pie).

7. **Cartesian extraction (T4):** The Cartesian handler executes seven internal stages: orientation validation → meta-clustering algorithm selection (HDBSCAN / DBSCAN / KMeans / intersection-alignment) → spatial label classification → dual-axis detection → calibration (weighted least squares; adaptive confidence from 0.8 down to 0.0; R² quality gates: CRITICAL_R2 = 0.85 warning, FAILURE_R2 = 0.40 hard-failure threshold) → baseline detection → chart-specific value extraction.

8. **Result formatting and persistence (T7):** The pipeline serializes results into a `PipelineResult` with provenance. Three artifact files are produced: `_consolidated_results.json`, `_protocol_export.csv`, and `_run_manifest.json`.

9. **Protocol row build, review, and CSV filtering (T5/T6/T7):** Protocol rows map extracted elements to a structured schema. In the GUI, the user can filter rows by outcome and group, edit values in-table (first edit snapshots the original value and marks `review_status = corrected`), and export a filtered CSV.

10. **Validation metrics and CI gates (T3–T8):** A built-in validation harness (`src/validation/run_protocol_validation.py`) computes success rate, categorical accuracy, Lin's CCC, and Cohen's Kappa by comparing predicted protocol CSVs against gold-standard CSVs. Gate thresholds define release readiness.

### Configurable Settings (T2)

The following parameters are configurable via CLI flags or the GUI settings panel:

| Parameter | Options | Default |
|---|---|---|
| Input type | `auto`, `image`, `pdf` | `auto` |
| OCR engine | `Paddle`, other supported engines | `Paddle` |
| OCR accuracy | `Optimized`, `Standard` | `Optimized` |
| Calibration method | `PROSAC`, others | `PROSAC` |
| DocLayout text extraction | `true`, `false` | `true` |
| Context file | `.json` path | None |
| Outcome filter | string | None |
| Group filter | string | None |
| Annotated output | flag | disabled |

---

## Dataset

Most articles used in this study originate from a systematic review investigating the effects of antidepressants on rodents in the forced swim test (Martins et al., 2025). This behavioural test measures immobility, swimming, and climbing responses as indicators of potential antidepressant effects in preclinical research. The dataset is divided into three samples:

**Sample I — Real-world scientific PDFs (primary validation)**

A sample of 200 articles was randomly drawn from 2,560 records using the "RANDBETWEEN" function in Microsoft Excel (v. 16.0, Microsoft Corp., Redmond, WA, USA). The publication date range spans 1986 to 2017, reflecting changes in reporting style over time. Of 561 studies within the 200 publications, 435 presented graphical data (bar plots: n = 423, 97.24%; line plots: n = 12, 2.76%). This sample is used to validate all eight functionalities (T1–T8) under default and customized settings.

**Sample II — Synthetically generated plots (in-depth extraction evaluation)**

Approximately 800 plots per chart type (bar, line, scatter, box, histogram, heatmap, area, and pie) were synthetically generated using a data generation pipeline that produces ground-truth labels (bounding boxes, keypoints, text content) simultaneously with the images, ensuring 100% annotation accuracy. Values are sampled from statistical distributions reported in the scientific literature and rendered with the Matplotlib library. An augmentation pipeline simulates scanning artifacts, low resolution, and sensor noise. The dataset spans 8 plot patterns, more than 20 visual themes, and more than 15 visual effects. Synthetic figures are embedded into real PDF files drawn from the remaining systematic review corpus to test plot detection (T3) and context filtering (T5) in realistic document layouts.

**Sample III — External benchmark (time and performance comparison)**

The Cramond et al. (2019) benchmark dataset of 23 plots was identified as a candidate for cross-tool time and accuracy comparison. *Note: at the time of protocol preparation, access to the original PDF or image files was not confirmed (permission request pending with the original authors). This sample will be included if access is granted prior to the start of testing; otherwise, the time comparison in T8 will rely exclusively on Samples I and II.*

---

## Requirements

### Functional Requirements

The tool must:
- Accept PDF files, image files, or directories of either type as input.
- Automatically detect, rasterize, and classify embedded plot images.
- Extract quantitative and descriptive data from all eight supported chart types.
- Support user-configurable OCR engine, calibration method, and model selection.
- Allow context-based filtering of outcomes and experimental groups.
- Enable in-interface review and correction of extracted values with full edit provenance.
- Export the final dataset as a structured `.csv` file and a `.json` manifest.

### Non-Functional Requirements

**Cross-platform compatibility:** The software must run on Windows, macOS, and Linux.

**Accessibility:** The tool is available for local installation and runs with a graphical user interface (`src/main_modern.py`) as well as a command-line interface (`src/analysis.py`), enabling use without programming experience.

**Workflow integration:** The tool is designed to be importable as a Python package for integration into automated Python or R pipelines (package metadata (`pyproject.toml`) is pending finalization).

**Documentation:** Comprehensive installation instructions, CLI/GUI usage guides, input/output specifications, and example workflows will be provided.

**Model distribution:** Detection and classification models are distributed as ONNX files via the installer manifest. Migration from the current Google Drive source to HuggingFace is planned for a future release.

---

## Known Technical Limitations at Validation Onset

The following limitations are confirmed from the current implementation and should be considered when interpreting validation results:

1. **Bar label association:** The metric-learning association path (`bar_association_mode='metric_learning'`) requires a trained `.npz` weight file not yet available; the heuristic path is the active default.
2. **Heatmap color calibration:** The CIELAB B-spline calibration mode (`heatmap_color_mode='lab_spline'`) is implemented but opt-in; the legacy HSV-based pipeline remains the default.
3. **Strategy router:** A multi-strategy dispatch layer (`StandardStrategy`, `VLMStrategy`, `HybridStrategy`) is implemented in `src/strategies/` but is not yet wired into the active pipeline. The standard extraction path is always used during this validation.
4. **Conformal prediction intervals:** Per-element uncertainty intervals (`src/calibration/conformal.py`) exist but require an offline calibration corpus step that has not yet been run; uncertainty fields will be absent from output during validation.
5. **Calibration hard-failure:** A calibration R² below 0.40 currently terminates extraction for the affected image (FAILURE_R2 hard-failure path in `src/handlers/base.py`). This will suppress rows for poorly calibrated charts and should be tracked in validation results.
6. **Package metadata:** `pyproject.toml`/`setup.py` are absent; installation must follow the documented manual environment setup until this is resolved.

---

## Validation Planning

### Test Planning

| Test ID | Functionality | Input | Expected Result | Acceptance Criteria |
|---|---|---|---|---|
| T1 | File import | Folder containing PDF files and/or image files | All assets loaded and rasterized successfully; provenance metadata attached to PDF-derived assets | No reading error (success rate > 99%) |
| T2 | Choice of settings | Interaction with CLI flags or GUI settings panel | Selected OCR engine, calibration method, and model loaded without error; settings exported | No errors during module load (success rate > 99%) |
| T3 | Plot and text recognition | PDF files (Samples I and II) | Correct identification and classification of plot types: a) bar, b) box, c) scatter, d) line, e) heatmap, f) histogram, g) area, h) pie; recognition of axis labels, titles, and embedded legends | Accuracy > 90%; F-score > 0.85 per type |
| T4 | Data extraction | Plots of all eight types (Samples I and II) | Extraction of legend, title, axis titles, and type-specific values: a) bar: scale, groups/categories, bar heights, error bars; b) box: scale, groups/categories, min, Q1, median, Q3, max, outliers; c) scatter: x-scale, y-scale, x-coordinates, y-coordinates, categories; d) line: x-values, y-values, groups/categories; e) heatmap: row/column categories, cell values; f) histogram: bin centers, bin edges, bin heights; g) area: x-values, y-values, AUC, baseline; h) pie: categories, proportions | Categorical variables: accuracy > 90%; Numerical variables: CCC > 0.90 |
| T5 | Context-of-interest filtering | `.json` context file + outcome/group filter inputs | Protocol CSV containing only rows matching the specified outcome and group filters | Accuracy > 90% |
| T6 | Correction of extracted data by the user | GUI protocol table (backed by in-memory `.json` protocol rows) | Edited values persisted in protocol rows; `review_status` set to `corrected`; original values preserved in `_original` snapshot | All corrections recorded without error; categorical accuracy > 90%; CCC > 0.90 post-correction |
| T7 | Export of results | Completed extraction and optional correction | `_protocol_export.csv` and `_consolidated_results.json` generated with all expected columns | No export errors (success rate > 99%) |
| T8 | Processing time | Timestamp at pipeline start | Timestamp at pipeline end for each sample | Total automated processing time shorter than manual extraction or WebPlotDigitizer (Cramond et al., 2019); statistical comparison using t-test (normal) or Wilcoxon test (non-normal) |

**Table 1.** Test summary for each tool functionality.

---

### Test Execution

#### Procedures — Sample I

The validation process will be performed twice: **(1)** with default settings and **(2)** with customized settings. All functionalities (T1–T8) will be tested for each configuration.

**i. File import (T1):**
A folder containing 200 PDF files will be provided as input via the CLI (`--input ./sample_I --input-type pdf`). The system will rasterize all embedded plot images and attach provenance metadata. The number of successfully loaded files and any reading errors will be recorded.

**ii. Choice of settings (T2):**
In round 1, default settings are used (PaddleOCR Optimized, PROSAC calibration). In round 2, the tester will manually select alternative settings through the GUI or equivalent CLI flags. The loaded module configuration will be verified and exported.

**iii. Plot and text recognition (T3):**
The system will classify each detected image into one of the eight chart types and extract embedded textual elements (axis labels, titles, legends). Results will be compared against a gold-standard annotation prepared by two independent human reviewers.

**iv. Data extraction (T4):**
For each plot, the system will extract numerical and descriptive elements as specified in Table 1. Extracted outputs will be compared against gold-standard values established by two independent human reviewers. Inter-rater reliability will be assessed prior to gold-standard finalization.

**v. Context-of-interest filtering (T5):**
A `.json` context file containing experimental group and outcome metadata will be provided (`--context ./context.json`). The CLI flags `--filter-outcome` and `--filter-group` (or equivalent GUI filter boxes) will be applied. The resulting protocol CSV will be verified to contain only the targeted rows.

**vi. Correction of extracted data by the user (T6):**
The tester will load the protocol table in the GUI and manually correct a predefined set of extracted values. The corrected protocol rows will be verified against the expected corrections, and the `review_status` field will be checked for `corrected` markings.

**vii. Export of results (T7):**
After extraction and optional correction, the export function will be executed. The resulting `_protocol_export.csv` and `_consolidated_results.json` files will be verified for completeness and structural integrity.

**viii. Time (T8):**
The start and end timestamps for the full pipeline run will be recorded (available in `_run_manifest.json`). Processing time will be compared with published manual extraction times and WebPlotDigitizer benchmarks from Cramond et al. (2019).

---

#### Procedures — Sample II

**Plot and text recognition (T3):**
Synthetic PDFs containing plots of all eight chart types will be processed. The system will classify each image and extract textual elements. Results will be compared against the 100%-accurate synthetic ground-truth labels.

**Context-of-interest filtering (T5):**
Synthetic metadata context files will be provided. The system will filter protocol rows by outcome and group. Results will be compared against the known ground truth.

**Data extraction (T4):**
For each synthetic plot, extracted numerical and categorical values will be compared against the synthetic ground-truth labels generated simultaneously with the images.

---

#### Procedures — Sample III *(conditional)*

*This sample will be included only if access to the Cramond et al. (2019) plot dataset is confirmed. If included:* the total processing time for all 23 plots will be recorded and compared with the extraction times reported in Cramond et al. (2019) for the "current method" and their new application.

---

### Register of Results

| Test ID | Expected Result | Obtained Result | Accordance | Observations |
|---|---|---|---|---|
| T1 | All assets loaded with no reading errors (success rate > 99%) | | | |
| T2 | Settings loaded without error (success rate > 99%) | | | |
| T3 | All 8 chart types correctly classified; textual elements extracted (accuracy > 90%) | | | |
| T4 | Type-specific values extracted (categorical accuracy > 90%; CCC > 0.90 for numerical) | | | |
| T5 | Protocol CSV contains only filtered outcome/group rows (accuracy > 90%) | | | |
| T6 | Corrections persisted; `review_status = corrected` confirmed (accuracy > 90%; CCC > 0.90) | | | |
| T7 | CSV and JSON artifacts generated without errors (success rate > 99%) | | | |
| T8 | Automated processing time shorter than manual/comparator tool | | | |

---

## Performance Metrics

Metrics will be calculated using the `{yardstick}` R package and/or the built-in Python validation harness (`src/validation/run_protocol_validation.py`).

**T1, T2, T7:**

\[ \text{Success Rate (\%)} = \frac{N_{\text{successful executions}}}{N_{\text{attempts}}} \times 100 \]

**T3 and T5** — Per chart type (T3) or per outcome/group (T5):

\[ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \]

\[ \text{F-score} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN} \]

\[ \text{Recall/Sensitivity} = \frac{TP}{TP + FN} \]

\[ \text{Specificity} = \frac{TN}{TN + FP} \]

\[ \text{Precision} = \frac{TP}{TP + FP} \]

**T4 and T6** — Per extracted field:
- *Descriptive/categorical data:* Accuracy, F-score, Recall/Sensitivity, Specificity, Precision.
- *Numerical data:* Lin's Concordance Correlation Coefficient (CCC).

**T8:**
The processing time difference will be expressed as a percentage relative to the comparator. Statistical comparisons will use a t-test (normally distributed data) or Wilcoxon signed-rank test (non-normal data).

**Inter-rater reliability (gold-standard annotation):**
- Categorical variables: Cohen's Kappa ≥ 0.81
- Numerical variables: Intraclass Correlation Coefficient (ICC) ≥ 0.81

---

## Success Criteria

| Metric | Threshold |
|---|---|
| File import success rate (T1) | > 99% |
| Settings load success rate (T2) | > 99% |
| Chart classification accuracy (T3) | > 90% |
| Data extraction — categorical accuracy (T4) | > 90% |
| Data extraction — numerical CCC (T4) | > 0.90 |
| Context filtering accuracy (T5) | > 90% |
| User correction fidelity (T6) | No data-loss errors; accuracy > 90% |
| Export success rate (T7) | > 99% |
| Processing time vs. comparator (T8) | Statistically significantly shorter |
| Inter-rater Kappa (gold-standard) | ≥ 0.81 |
| Inter-rater ICC (gold-standard) | ≥ 0.81 |

---

## Qualitative Survey

Testers will respond to the following items using a 5-point Likert scale (−2 Strongly disagree, −1 Disagree, 0 Neutral, +1 Agree, +2 Strongly agree), plus one open-ended item:

1. The interface and layout are intuitive.
2. The instructions and error messages were clear.
3. With more experience, the tool becomes easier to use.
4. The tool processed the files in an adequate time.
5. The tool did not experience any crashes or slowdowns.
6. The tool provides all the functionalities necessary for extracting data from plots.
7. I trust the results provided by the tool.
8. I would use the tool in future projects.
9. I would recommend the tool to colleagues.
10. The setting and customization options are comprehensive enough for my needs.
11. *(Open-ended)* Any additional comments or suggestions?

---

## References

Bannach-Brown, A., Hair, K., Bahor, Z., Soliman, N., MacLeod, M., & Liao, J. (2021). Technological advances in preclinical meta-research. *BMJ Open Science, 5*(1). https://doi.org/10.1136/bmjos-2020-100131

Bugajska, J. V., Hild, B. F., Brüschweiler, D., Meier, E. D., Bannach-Brown, A., Wever, K. E., & Ineichen, B. V. (2025). How long does it take to complete and publish a systematic review of animal studies? *BMC Medical Research Methodology, 25*(1), 226. https://doi.org/10.1186/s12874-025-02672-5

Cramond, F., O'Mara-Eves, A., Doran-Constant, L., Rice, A. S., Macleod, M., & Thomas, J. (2019). The development and evaluation of an online application to assist in the extraction of data from graphs for use in systematic reviews. *Wellcome Open Research, 3*, 157. https://doi.org/10.12688/wellcomeopenres.14738.3

Ioannidis, J. P. A. (2023). Systematic reviews for basic scientists: a different beast. *Physiological Reviews, 103*(1), 1–5. https://doi.org/10.1152/physrev.00028.2022

Martins, T., Ramos-Hryb, A. B., da Silva, M. A. B., do Prado, C. S. H., Eckert, F. B., Triches, F. F., da Costa, J. E., Bolzan, J. A., McCann, S. K., & Lino de Oliveira, C. (2025). Antidepressant effect or bias? Systematic review and meta-analysis of studies using the forced swimming test. *Behavioural Pharmacology, 36*(6), 347–363. https://doi.org/10.1097/FBP.0000000000000844

Van der Mierden, S., Spineli, L. M., Talbot, S. R., Yiannakou, C., Zentrich, E., Weegh, N., Struve, B., Zur Brügge, T. F., Bleich, A., & Leenaars, C. H. C. (2021). Extracting data from graphs: A case-study on animal research with implications for meta-analyses. *Research Synthesis Methods, 12*(6), 701–710. https://doi.org/10.1002/jrsm.1481
