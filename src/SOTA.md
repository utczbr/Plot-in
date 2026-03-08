## High‑level Phase 1 takeaways

Modern chart systems cluster into two families:

- **Structure‑first pipelines** (ChartOCR‑style) that do detection/OCR/geometry then derive values.  
- **VLM / chart‑foundation models** (MatCha, DePlot, UniChart, ChartVLM, TinyChart) that map images (plus text) directly to tables or answers. [aclanthology](https://aclanthology.org/2023.findings-acl.660/)

Recent work in documents and OCR introduces **hybrid VLM pipelines** where detection/layout modules run first and a VLM is used as a semantic formalizer or fallback, which closely matches your proposed `StandardStrategy` / `VLMStrategy` / `HybridStrategy` architecture. [arxiv](https://arxiv.org/html/2601.21639v1)

For **uncertainty**, the most relevant SOTA is **conformal prediction**: a model‑agnostic post‑hoc layer that wraps any predictor (regression, detection, segmentation) with per‑instance intervals or masks that have guaranteed coverage, including for bounding boxes in vision. This is a good fit for replacing your ad‑hoc R² and “perturb at pixel 100” logic and for driving confidence‑based routing. [arxiv](https://arxiv.org/abs/2306.00876)

Below I break this down by (1) SOTA chart/VLM models, (2) hybrid routing patterns, (3) conformal/uncertainty methods, and (4) how they address your three target issues: monolithic pipeline, R²<0.40 hard fail, and lack of confidence‑based fallback—always mapped back to your current contracts.

***

## SOTA chart and VLM models relevant to routing

### DePlot and MatCha (Google)

MatCha is a pixels‑to‑text foundation model trained on **chart de‑rendering and math reasoning**, starting from Pix2Struct, with pretraining tasks for plot deconstruction and numerical reasoning; it significantly outperforms prior ChartQA/PlotQA models. [aclanthology](https://aclanthology.org/2023.acl-long.714/)

DePlot is a **modality conversion module** built on top of MatCha that translates a chart image into a linearized table, which is then passed to an LLM (e.g., Flan‑PaLM) for reasoning, and is explicitly positioned as a **plug‑and‑play chart‑to‑table component**. [arxiv](https://arxiv.org/abs/2212.10505)

**Integration pattern for you**

- DePlot/MatCha naturally implement your planned **`ChartToTableStrategy`**:
  - `PipelineStrategy.execute` calls DePlot with the raw image and chart type, gets a table, and converts it into your `ExtractionResult.elements` schema (groups/values, units, baselines when derivable).
  - Because DePlot is already “chart‑to‑table”, it can bypass **detection/doclayout/orientation/OCR** stages without needing to change their existing implementations; this matches your idea that `StandardStrategy` wraps current stages 3–7 while `ChartToTableStrategy` directly returns an `ExtractionResult`.
- **Contracts preserved**:
  - You keep `PipelineResult` unchanged; only `result.diagnostics['strategy_id'] = 'chart_to_table'` and maybe `value_source='chart_to_table'` are added as *optional* keys, which your docs already reserve as additive fields.
  - PDF provenance, protocol rows, and CSV export continue to work because they operate on `ExtractionResult`/`PipelineResult`, not on how values were obtained.

***

### UniChart

UniChart is a **universal chart pretrained model** that encodes chart text, data, and visual elements, then uses a chart‑grounded text decoder, with pretraining tasks for both low‑level element extraction and high‑level reasoning. [chatpaper](https://chatpaper.com/chatpaper/paper/11844)

It explicitly models chart structure (bars, lines, text, data) and achieves SOTA across ChartQA, Chart‑to‑Text, Chart‑to‑Table, and related benchmarks, with good generalization to unseen chart styles. [openreview](https://openreview.net/forum?id=4MjZNeTCqZ)

**Integration pattern**

- UniChart is good as a **`VLMStrategy` backend** focused on reasoning and flexible question answering:
  - For pipeline mode `pipeline_mode='vlm_chart'`, route images of supported chart types to UniChart, ask for a structured JSON table or key‑value description instead of free‑form text, then map that into your `elements` list.
  - Because UniChart encodes chart elements explicitly, you can also experiment with **HybridStrategy** where you pass both the **original image and your detected elements/axis labels as text prompts**—leveraging your existing detection/OCR while letting UniChart reason over them.

Again, only strategy choice changes; `PipelineResult` and protocol remain intact.

***

### ChartVLM and ChartX

ChartVLM is a chart‑specialized MLLM trained and evaluated on ChartX, a benchmark covering **18 chart types and 7 tasks** (QA, captioning, etc.). [arxiv](https://arxiv.org/abs/2402.12185)

Experiments show ChartVLM matches or surpasses general‑purpose MLLMs (including GPT‑4V) on chart reasoning while remaining more interpretable and chart‑aware. [github](https://github.com/Alpha-Innovator/ChartVLM)

**Integration pattern**

- ChartVLM is another strong **`VLMStrategy` candidate**, particularly if you want broad chart‑type coverage aligned with your 8‑chart registry:
  - Implement `VLMStrategy` with a backend abstraction (e.g., `ChartVLMBackend`, `UniChartBackend`, `TinyChartBackend`), selected via configuration, so your router can swap models without contract changes.
  - Use ChartX’s multi‑task design to inspire your **quality metrics**: you can log per‑strategy success rate and accuracy by chart type, paralleling ChartX’s evaluation setup, and feed that into your Isolation‑First A/B gates.

***

### TinyChart and other efficient MLLMs

TinyChart is a 3B‑parameter chart MLLM that combines **Program‑of‑Thoughts (PoT) learning for numerical computation** with **visual token merging** for efficient high‑resolution encoding, achieving SOTA across ChartQA, Chart‑to‑Text, Chart‑to‑Table, OpenCQA, and ChartX. [emergentmind](https://www.emergentmind.com/papers/2404.16635)

Its PoT strategy explicitly has the model emit Python code to do calculations, reducing numeric errors that plague many VLMs on chart tasks. [arxiv](https://arxiv.org/abs/2404.16635)

**Integration pattern**

- TinyChart is particularly attractive for **resource‑bounded VLMStrategy**:
  - It can run as the default `VLMStrategy` for on‑prem / CPU‑constrained deployments, with larger models (ChartVLM, GPT‑4V) reserved for offline or high‑latency paths.
  - Its PoT style aligns with your **side‑channel diagnostics**: you can dump the generated Python code into `diagnostics['vlm_computation_trace']` without affecting `elements`.

***

## Hybrid routing patterns in recent work

### ChartOCR: deep hybrid framework

ChartOCR proposes a **deep hybrid framework**: keypoint detection and chart‑type classification as a shared stage, followed by **type‑specific rules** to build components and a data‑range extraction step to map pixel coordinates to numeric values. [microsoft](https://www.microsoft.com/en-us/research/publication/chartocr-data-extraction-from-charts-images-via-a-deep-hybrid-framework/)

This shows a successful pattern of **common neural stages + chart‑specific geometric/rule‑based post‑processing**, which is analogous to your shared 7‑stage Cartesian pipeline plus per‑type extractors.

**Relevance to your routing**

- ChartOCR’s architecture supports the idea that **shared upstream detection/OCR + per‑type logic** is robust; you can preserve your existing Standard path and add parallel strategies rather than replacing it wholesale.
- The paper’s emphasis on keypoint‑based deconstruction and data‑range extraction suggests a natural division: your **StandardStrategy** remains detection/OCR/baseline‑centric, while **VLM/ChartToTable strategies** handle cases where chart structure is intact but calibration or OCR is unreliable.

***

### OCRVerse and VLM‑based hybrid OCR pipelines

OCRVerse, a holistic OCR framework, explicitly distinguishes **traditional pipelines** (layout detection + OCR + heuristics) from **VLM‑based pipelines** that use detectors to crop and order regions, then feed them to a VLM. [arxiv](https://arxiv.org/html/2601.21639v1)

The paper positions VLM‑based pipeline methods as a **hybrid solution** that integrates explicit layout priors with LVLM semantics, and cites systems like MinerU and PaddleOCR‑VL, which keep classic detectors but use a VLM for higher‑level understanding and generation. [arxiv](https://arxiv.org/html/2601.21639v1)

**Relevance**

- This directly supports your **`HybridStrategy`** concept:
  - Run your full Standard pipeline to obtain elements, labels, calibration, and baselines.
  - Pass the image plus a structured summary of these (e.g., bounding boxes + recognized text serialized as JSON or prompt) into a VLM, asking it to validate or correct the extracted table.
- OCRVerse also underscores that **VLM‑only pipelines can be slow and costly**, so hybrid strategies are common for production scenarios—aligning with your need to keep latency within budget.

***

### VLM‑as‑formalizer and “Can VLMs replace OCR pipelines?”

The “VLM‑as‑Formalizer Pipelines” survey describes an architectural pattern where a VLM converts multimodal inputs into **structured formal outputs** (graphs, tables, code) that replace much of a multi‑stage OCR/detection pipeline, either as a single‑step system or as one module in a hybrid pipeline. [emergentmind](https://www.emergentmind.com/topics/vlm-as-formalizer-pipelines)

A retail case study on replacing OCR‑based VQA pipelines with VLMs concludes that while VLMs can simplify architectures, they often require **careful gating and sometimes fallback to classical OCR components** for reliability and cost reasons. [arxiv](http://arxiv.org/html/2408.15626)

**Relevance**

- These works validate your design of:
  - A **pure VLM path** (`VLMStrategy` or `ChartToTableStrategy`) that bypasses detection/doclayout/OCR.
  - A **router that can still fall back** to the “old” Standard path for cost/latency or safety reasons.
- They also justify your plan to expose **`strategy_id` and `strategy_confidence`** in `diagnostics` so downstream consumers can treat VLM‑derived outputs differently if desired.

***

### Chart VLM robustness and the need for routing

Recent evaluations like “Do VLMs really understand charts?” and “Are Large Vision Language Models up to the Challenge of Chart Comprehension and Reasoning?” show that chart VLMs can produce fluent answers but suffer from **robustness, consistency, and hallucination issues**, especially on harder questions and distribution shifts. [aclanthology](https://aclanthology.org/2024.findings-emnlp.973/)

These studies recommend **task‑aware evaluation and robustness checks**, highlighting that VLMs should often be combined with structural priors or additional verification if used in safety‑critical chart reasoning. [bohrium.dp](https://bohrium.dp.tech/paper/arxiv/2406.00257)

**Relevance**

- These findings strongly support your **Isolation‑First** and **HybridStrategy** approach:
  - You should not make `VLMStrategy` the default until you have protocol‑level evidence; instead, run A/B via your validation harness and use protocol metrics (success rate, CCC, Kappa) as gates.
  - When Hybrid is active, you can treat VLM outputs as **validation / correction layers** on top of Standard results instead of hard‑replacing them, consistent with these papers’ caution about hallucinations.

***

## Conformal prediction and uncertainty quantification

### Conformal prediction basics

Conformal prediction (CP) is a post‑hoc method that wraps any predictive model with **per‑instance prediction sets or intervals** that guarantee a chosen coverage level (e.g., 95%) under mild assumptions, without requiring Bayesian models or distributional assumptions. [arxiv](https://arxiv.org/pdf/2304.06052.pdf)

Recent work proposes methods for **quantifying model uncertainty from CP prediction sets**, comparing CP‑derived uncertainty against Bayesian and ensemble methods and giving certified bounds on uncertainty measures. [arxiv](https://arxiv.org/abs/2306.00876)

**How this replaces your current ad‑hoc uncertainty**

- Instead of:
  - R² thresholds only.
  - A fixed pixel perturbation at 100 px.
  - 1.96σ Gaussian intervals from residuals.
- You can:
  - Build CP calibrators for each **chart type + value family** (bar heights, scatter points, box stats, heatmap cell values, etc.) using your existing validation corpus.
  - For a new extraction, compute a non‑conformity score (e.g., absolute or relative error proxy) for each value and return a **per‑value interval** with guaranteed coverage at your chosen level.

These CP intervals provide a principled `uncertainty` field that is model‑agnostic and works with both Standard and VLM strategies.

***

### CP for vision and detection tasks

Several recent works extend CP to **vision problems**:

- **Conformal prediction masks** provide per‑pixel uncertainty in image‑to‑image tasks (colorization, completion, super‑resolution), yielding interpretable masks that highlight high‑uncertainty regions while giving formal risk guarantees. [openreview](https://openreview.net/pdf?id=J4QatK02Qc)
- For **object detection**, “Adaptive Bounding Box Uncertainties via Two‑Step Conformal Prediction” and related works derive **class‑conditional bounding‑box intervals** with guaranteed coverage and adapt interval sizes to object size using ensembles and quantile regression. [alextimans.github](https://alextimans.github.io/docs/conformalbbox_uncv23.pdf)

**Relevance to chart extraction**

- Charts are essentially **structured detection + regression**:
  - You detect visual elements (bars, lines, points, cells).
  - You regress values (via calibration) per element.
- You can adapt detection‑oriented CP to charts by:
  - Treating each extracted value as a regression output.
  - Optionally also calibrating **bounding‑box uncertainty** for detections (e.g., bar endpoints), but for protocol you mainly care about numeric intervals.

Your Critic.md already proposes conformal prediction for uncertainty; these papers give you ready‑made scoring and calibration patterns, especially for handling heteroskedastic errors (e.g., larger bars having larger absolute error), as in size‑adaptive box intervals. [eccv.ecva](https://eccv.ecva.net/virtual/2024/poster/138)

***

## Handling uncalibrated and single‑axis charts

### How SOTA systems avoid hard failures

Structure‑first systems like ChartOCR **never hard‑fail solely on missing calibration**; instead they either:

- Fall back to **relative or percentage‑based outputs** (e.g., pie charts always sum to 100%) or
- Use detected range labels and OCR’d numbers to derive mappings and only abstain or degrade outputs when essential labels are missing. [microsoft](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/12/WACV_2021_ChartOCR.pdf)

VLM/Chart foundation models (MatCha, DePlot, UniChart, ChartVLM, TinyChart) generally do **not explicitly compute an R² between pixel positions and axis values**. Instead they learn an implicit mapping from image (plus axis ticks/text) to data, and when axis labels are absent they can still often answer relative or qualitative questions, albeit with degraded accuracy. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2402.12185)

**Implication**: it is unusual in SOTA chart work to declare complete failure purely because calibration is weak; more commonly, systems degrade gracefully (e.g., relative values, percentages, or abstentions on precise numeric values).

***

### Single‑axis and partial calibration

UniChart’s architecture explicitly encodes both **visual elements and associated text and data**, which lets it handle cases where one axis is less informative (e.g., qualitative x‑axis categories and numeric y‑axis). [aclanthology](https://aclanthology.org/2023.emnlp-main.906/)

MatCha and DePlot’s chart‑to‑table task similarly trains on varied chart styles where one axis may be categorical and only the other numeric, forcing the model to treat **partial calibration as normal** rather than an error. [aclanthology](https://aclanthology.org/2023.findings-acl.660/)

**How that translates to your R² gate**

- Instead of treating **R² < 0.40 as fatal** for all Cartesian charts, SOTA practice suggests:
  - Allow **pixel‑only or partially calibrated extraction** with explicit flags.
  - Escalate to a **VLM / ChartToTable strategy** that does not depend as heavily on R², particularly when text labels or numeric data labels are available.

This dovetails with your planned “approximate” and “uncalibrated” calibration_quality levels and the idea of VLM validation.

***

## Concrete integration proposals mapped to your architecture

Below I map the SOTA ideas directly onto your existing components, explicitly preserving contracts.

### 1. StrategyRouter‑driven hybrid pipeline

**Design inspired by DePlot / MatCha / UniChart / TinyChart / ChartVLM + OCRVerse/VLM‑as‑formalizer**

- Keep your existing **8‑stage pipeline** as the implementation of `StandardStrategy.execute(image, chart_type, conf, services)`.
- Implement additional strategies:

  1. `ChartToTableStrategy` (DePlot/MatCha backend).  
     - Input: `image`, `chart_type`.  
     - Output: `ExtractionResult` where `elements` is derived from the returned table.  
     - Set `diagnostics['strategy_id']='chart_to_table'`, `value_source='chart_to_table'`.

  2. `VLMStrategy` (UniChart / ChartVLM / TinyChart backend).  
     - Input: `image`, `chart_type`, plus optionally a JSON summary of detections/OCR (Hybrid flavor).  
     - Output: `ExtractionResult` mapped from structured JSON produced by the VLM.  
     - Set `strategy_id='vlm'`, potentially `diagnostics['vlm_model']='unichart'` etc.

  3. `HybridStrategy`.  
     - Runs `StandardStrategy` first and computes **quality scores** from:
       - Calibration R² and trivial‑calibration flag.
       - Conformal prediction interval widths (see below).
       - Detection/OCR confidences and missing essential components (e.g., no ticks but many data labels).
     - If scores indicate low quality, it calls either:
       - `VLMStrategy` as a full replacement, or
       - `VLMStrategy` as a **validator/corrector**: compare VLM‑derived table with Standard table, then:
         - Keep Standard when they agree within CP intervals.
         - Mark elements with `value_source='vlm_override'` where VLM is more confident.
     - Set `strategy_id='hybrid'`, `fallback_triggered=True`, and `standard_rejected_reason` accordingly.

**Contracts**

- `ChartAnalysisPipeline.run()` changes only in that it calls `StrategyRouter.select(...)` and then `strategy.execute(...)`, but it still returns exactly the same `PipelineResult`.
- You only add optional `diagnostics` keys, consistent with your “non‑breaking extensions” list.
- `HandlerContext` remains unchanged for StandardStrategy; `VLMStrategy` and `ChartToTableStrategy` can ignore `detections` and `chart_elements` by passing minimal contexts to handlers or bypassing handlers entirely and directly forming `ExtractionResult` objects (which your Critic already allows for non‑detection strategies).

***

### 2. Replacing R² hard‑fail with CP‑driven calibration quality

**Design inspired by conformal prediction in regression and detection** [arxiv](http://arxiv.org/pdf/2403.07263v2.pdf)

- On your **validation corpus**, for each chart type and extraction mode (e.g., bar, line, scatter, box, heatmap):

  1. Run your current Standard pipeline and log:
     - Predicted values \(\hat{y}_i\),
     - Ground truth \(y_i\),
     - Any relevant covariates (e.g., bar height in pixels, axis scale range).
  2. Define a **non‑conformity score** \(s_i\), e.g.:
     - \(s_i = |\hat{y}_i - y_i|\) or \(|\hat{y}_i - y_i| / (|y_i| + \epsilon)\).
  3. Use split CP to compute the empirical quantile \(q_\alpha\) of \(s_i\) for a desired coverage level \(1-\alpha\).

- At runtime, for each extracted value \(\hat{y}\):

  - Compute an interval \([\hat{y} - q_\alpha, \hat{y} + q_\alpha]\) (or a relative version).
  - Attach this as `uncertainty={'method': 'conformal', 'alpha': alpha, 'interval': [lo, hi]}` to the element.

- To handle **heteroskedasticity**, you can condition scores on features (e.g., bin the values by magnitude or axis range) as done in adaptive CP for detection, where interval widths adapt to object size. [eccv.ecva](https://eccv.ecva.net/virtual/2024/poster/138)

**Routing and calibration quality**

- Replace the current binary R² gate with a CP‑aware calibration quality:

  - If R² is high **and** CP intervals are tight relative to value ranges → `calibration_quality='high'`.
  - If R² is moderate and intervals moderate → `calibration_quality='approximate'`.
  - If R² is low or undefined and intervals wide → `calibration_quality='uncalibrated'` but you **still return values with intervals**, instead of failing.
- Use these quality levels inside `HybridStrategy`:
  - Automatically escalate to VLM/ChartToTable when `calibration_quality='uncalibrated'` *and* downstream protocol consumers require precise numeric values.

**Contracts**

- `ExtractionResult` gains optional per‑element uncertainty fields and a `calibration_quality` diagnostic, as you already propose; no schema break.
- Your protocol CSV can remain unchanged initially; if you later add CI columns, do so as optional columns at the end.

***

### 3. Confidence‑based routing (classification, detection, and VLM)

Building on ideas from **selective prediction** and CP‑based risk control plus empirical findings on VLM robustness for charts, you can define a **confidence score** that combines: [aclanthology](https://aclanthology.org/2024.findings-emnlp.973/)

- Classification confidence (already available).
- Detection coverage (number of detected bars/points vs expected).
- Calibration quality (R² + CP intervals).
- VLM self‑reported confidence where available (e.g., logprobs or explicit answer‑certainty prompts).

**Router policy examples**

- For low classification confidence (e.g., 0.2–0.4) **and** sparse detections:
  - Route directly to `VLMStrategy` or `ChartToTableStrategy`, since the Standard pipeline is likely to fail.
- For high classification/detection confidence but **low calibration quality**:
  - Run `StandardStrategy` for structure (grouping, labels) but call `VLMStrategy` to refine numeric values or to answer queries that require high precision, using CP intervals to determine when to trust the Standard values.
- For cases where VLMs are known to be brittle (from ChartX or EMNLP chart robustness papers) on certain chart types or tasks, you can invert the routing and prefer StandardStrategy unless Standard clearly signals low quality. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2402.12185)

All of this happens inside `StrategyRouter.select(...)`, which you already plan to use for **confidence‑based routing** and configuration overrides.

***

## How these address your three target problems

1. **Monolithic sequential pipeline**

   - SOTA suggests multiple **parallel strategies** (structure‑first, chart‑to‑table, VLM formalizer) rather than a single monolithic flow, matching your StrategyRouter architecture inspired by DePlot/MatCha (chart‑to‑table), UniChart/ChartVLM/TinyChart (VLM), and OCRVerse/VLM‑as‑formalizer pipelines. [aclanthology](https://aclanthology.org/2023.acl-long.714/)
   - Implementing `StandardStrategy`, `VLMStrategy`, `ChartToTableStrategy`, and `HybridStrategy` lets you keep the existing flow intact while adding SOTA alternatives behind feature flags.

2. **Fatal R² < 0.40**

   - Conformal prediction literature provides a **principled replacement for R²‑only gating**, giving per‑value intervals and enabling graceful degradation instead of hard failure. [openreview](https://openreview.net/pdf?id=J4QatK02Qc)
   - You can keep R² as a diagnostic but rely on CP interval width and coverage as your primary uncertainty metric, mapping directly into your proposed `calibration_quality` categories and avoiding unnecessary pipeline aborts.

3. **No confidence‑based fallback**

   - Chart robustness evaluations and VLM‑pipeline studies argue for **confidence‑ or quality‑aware routing**, particularly when VLM outputs can hallucinate or when classical OCR/detection struggles. [bohrium.dp](https://bohrium.dp.tech/paper/arxiv/2406.00257)
   - By combining classification confidence, detection coverage, calibration quality, and CP‑based uncertainty into a routing score, your `StrategyRouter` can:
     - Prefer StandardStrategy when it is strong and cheap.
     - Escalate to VLM/ChartToTable when Standard is weak.
     - Use HybridStrategy when both are available and you want cross‑verification.

All of these changes are compatible with your Isolation‑First policy: run A/B experiments by toggling `pipeline_mode`, keep protocol contracts identical, and require gate metrics (success rate, accuracy, CCC, Kappa) to meet or exceed baseline before changing defaults.


Phase 2 SOTA guidance for your Cartesian extractors:

Use learned grouping/association and keypoint‑style value extraction where possible, and reserve simple normalized heuristics for cheap, high‑precision fallbacks. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/1906.11906)

***

## Cross‑cutting Cartesian upgrades

Your current Cartesian handlers share strong structural patterns but rely on many absolute pixel thresholds and a hard R² gate, which makes them brittle across resolutions and chart styles. A consistent SOTA direction is to (1) normalize everything to image or element scale, (2) push association and validation into **learned scoring models**, and (3) use conformal prediction (Phase 1) for numeric uncertainty rather than magical constants. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77311235/c0a42b95-945e-4507-a949-047171a75c05/paste.txt)

Concrete upgrades (shared across bar/box/hist/hist/scatter/line):

- Replace absolute thresholds with **resolution‑normalized and element‑normalized rules** (e.g., distances and widths as fractions of max(image_h, image_w) and local bar/marker sizes). This aligns with modern chart frameworks (ChartEye, ChartReader) which treat sizes and gaps relative to chart layout rather than pixels. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2408.16123)
- Integrate the conformal layer from Phase 1 into all Cartesian extractors so that every numeric output has an interval and your handlers can emit `calibration_quality` instead of hard‑failing on R². [arxiv](https://arxiv.org/abs/2306.00876)
- Centralize robust numeric parsing (Unicode minus, thousands separators, percentages) in a utility module so all calibration and data‑label overrides reuse the same parser; this is exactly the failure mode observed in materials‑science chart extraction where axis/legend text is diverse and noisy. [openreview](https://openreview.net/pdf?id=vj8dqNrzEe)

***

## Bar charts: layout, values, association

Your bar path currently uses 20+ hardcoded thresholds, a 2.5× inter‑bar spacing heuristic for layout, a four‑tier handwritten label association, and endpoint‑based value computation. SOTA bar/plot systems instead use **deep matching modules** and structured detection to learn bar–label associations and value extraction jointly. [arxiv](https://arxiv.org/pdf/1906.11906.pdf)

### Layout detection (grouped vs clustered vs single)

- ChartReader and the “single deep neural network” model both treat bar charts as a detection + grouping problem, using bounding boxes and relative positions to infer bar groupings and x‑axis categories rather than global spacing heuristics. [cs.odu](https://www.cs.odu.edu/~jwu/downloads/pubs/rane-2021-iri/rane-2021-iri.pdf)
- Replace your `max_spacing > 2.5 × min_spacing` rule with:
  - Compute **all inter‑bar horizontal gaps**.
  - Fit a **1D Gaussian mixture (e.g., 1 vs 2 components)** to the gap distribution, using BIC/AIC to select whether there is a distinct “group gap” vs “within‑group gap”.
  - If a two‑component model is preferred and the ratio of means is above a configurable threshold, treat the larger cluster as inter‑group gaps and segment accordingly.

You can implement this in `bar_associator.py` while still exposing the same `group_id` fields to downstream protocol builders.

### Robust value computation

- Deep bar‑chart models estimate values from **top bar boundaries and axis tick labels** via learned regression rather than a single endpoint heuristic. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/1906.11906)
- You can:
  - Keep your calibrated baseline but compute **area‑aware estimates**: integrate the calibrated value along vertical samples within each bar (or average multiple height samples) to reduce sensitivity to noisy baseline and jagged bar tops.
  - Use your conformal layer to return an interval instead of a single point when calibration is weak; this mirrors how line‑chart digitizers report uncertainty bands. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1524070325000062)

This keeps your `value` field but adds `uncertainty` and `value_source='calibrated_height'`.

### Learned bar–label association (P2 metric‑learning replacement)

The “Data Extraction from Charts via Single Deep Neural Network” framework adds an **object‑matching branch** that pairs detected bar boxes with text elements (x‑tick labels or legend entries), learning association as a classification task on pairwise features rather than hand‑tuned thresholds. [openreview](https://openreview.net/pdf?id=8nu1AqmMhO)

You can mirror this without changing your contracts:

- Build a **LightGBM or small MLP classifier** that scores (bar, text) pairs with features you already compute:
  - Normalized horizontal and vertical distances, overlap, alignment, color similarity, and cluster membership.
- Train it on your protocol gold data (bar ↔ label pairs), using “no‑match” negatives.
- Replace the current 4‑tier heuristic in `bar_associator.py` with:
  - Enumerate candidate pairs within a generous distance window.
  - Score with the learned model and take an assignment via Hungarian matching.
  - Emit the same `associated_label_id` fields.

This directly removes many magic numbers from your associator while following the same “single network with object‑matching branch” idea. [arxiv](https://arxiv.org/pdf/1906.11906.pdf)

### Learned error‑bar validation

Your error‑bar validator currently hardcodes multiple aspect/size/alignment weights. In ChartReader and related work, data quality is evaluated via **task‑specific learned metrics** such as F1 on bar–legend association and value accuracy. [cs.odu](https://www.cs.odu.edu/~jwu/downloads/pubs/rane-2021-iri/rane-2021-iri.pdf)

- Assemble a training set where each detected error bar is labeled as “valid”/“invalid” based on protocol gold.
- Use a small classifier over features like:
  - Bar height ratio, cap width normalized to bar width, vertical alignment with bar top, and distance to bar.
- Replace the six hand‑tuned thresholds with this classifier’s probability, emitting `error_bar_confidence` and a binary inclusion decision.

***

## Box plots: constrained five‑number summary and outliers

Your box extractor leaves invalid five‑number orderings uncorrected and lacks an outlier validation gate; whiskers are always 1.5×IQR even when data labels or clear outlier clusters suggest otherwise. SOTA chart‑digitization and curve‑extraction work highlights **explicit constraints and robust statistics** as key to reliable scientific data extraction. [dl.acm](https://dl.acm.org/doi/abs/10.1016/j.gmod.2025.101259)

### Enforcing valid five‑number summaries (P0)

- Before emitting min/Q1/median/Q3/max:
  - Sort the five values you inferred and project them onto a **monotone sequence**, e.g., by simply sorting or by a constrained optimization that minimally perturbs them.
- Emit the corrected sequence, plus a diagnostic like `diagnostics['five_number_corrected']=True` so downstream can audit.

This ensures no impossible box summaries reach protocol rows without altering the contract shape.

### Outlier validation gate (P0)

- Define whisker_low/high from the corrected min/max, then drop any “outliers” that fall inside \([whisker_low, whisker_high]\).
- For charts with numeric labels near points, give **precedence to labeled values** over geometric outlier estimates, as done in scientific line‑chart digitizers where data labels trump inferred points. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1524070325000062)

You can surface all kept outliers with `value_source='outlier_geometric'|'outlier_labeled'`.

### Ensemble median and adaptive whiskers (P1)

- Inspired by LineEX and AI‑ChartParser, which use multiple keypoint cues and curve pivots to localize line points robustly, run **all your existing median strategies in parallel** and combine them with confidence weights rather than a fixed cascade. [shiva-sankaran.github](https://shiva-sankaran.github.io/assets/docs/lineex.pdf)
- Derive whiskers adaptively:
  - If explicit point markers or data labels exist beyond 1.5×IQR, extend whiskers outward to cover them and mark IQR rule as secondary.
  - Only fall back to 1.5×IQR when there are no reliable outlier markers.

All of this is internal to `box_extractor.py` and keeps your current outputs while reducing brittle behavior.

***

## Histograms: orientation, bin contiguity, and semantics

You currently use a very simple `avg_height > avg_width` rule for histogram orientation, lack bin contiguity checks, and do not distinguish bin edges vs centers. SOTA chart frameworks treat histograms as a special case of **bar‑like detection plus instance‑level bin interpretation**, and they emphasize alignment between bars and text labels. [microsoft](https://www.microsoft.com/en-us/research/publication/chartocr-data-extraction-from-charts-images-via-a-deep-hybrid-framework/)

### Orientation parity with bar (P0)

- Reuse your **OrientationDetectionService** that you already apply to bar charts for histograms, so orientation leverages the same multi‑signal logic (variance, aspect, spatial cues) instead of a single height/width rule. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77311235/c0a42b95-945e-4507-a949-047171a75c05/paste.txt)
- Only fall back to `avg_height > avg_width` when orientation service fails or is disabled.

### Bin contiguity validation (P1)

- After detection and calibration, sort bins by x position and check:
  - Gaps larger than, say, 10–15% of median bin width (normalized) as **missing bins**.
  - Overlaps larger than the same threshold as **bin overlap anomalies**.
- Emit this in `diagnostics['bin_contiguity']='ok'|'gaps'|'overlaps'` and set a warning flag; this mirrors how scientific line‑chart digitizers validate sampling density and continuity. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1524070325000062)

### Bin edge vs center semantics (P2)

- Follow the heuristic you already sketched in Critic.md:
  - If `len(labels) == len(bars) + 1`, treat labels as **edges** and derive bin centers.
  - If `len(labels) == len(bars)`, treat labels as **centers**.
- When values are ambiguous, defer to VLM/ChartToTable strategies or mark bins with `value_source='ambiguous_bin_labels'` so protocol consumers can flag them.

***

## Scatter: sub‑pixel refinement and robust stats

Your scatter extractor assumes dark‑on‑light markers with Otsu, has a baseline sign bug and a dual‑axis safety net that aliases calibrations. Recent line/scatter digitization systems (LINEEX, AI‑ChartParser, AI‑ChartParser‑like work) show strong gains from **transformer‑based keypoint extraction** and from treating data extraction as instance segmentation or keypoint detection rather than pure centroiding. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70146?af=R)

### Baseline and dual‑axis fixes (P0)

- Normalize both axes to `value = pixel - baseline_pixel` to remove the sign inconsistency and ensure your calibration fits can be reused uniformly. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77311235/c0a42b95-945e-4507-a949-047171a75c05/paste.txt)
- For dual‑axis charts, if one axis calibration fails, **emit `None` and a warning** instead of copying the other axis’s calibration; this matches best practices in chart‑digitization tools that abstain rather than hallucinate values when an axis is missing. [openreview](https://openreview.net/pdf?id=vj8dqNrzEe)

You already have this planned; implementing it removes a major correctness bug without any schema change.

### Multi‑strategy sub‑pixel refinement (P1)

Borrow from keypoint‑centric methods like LINEEX and AI‑ChartParser:

- For each detected marker, compute three independent sub‑pixel centers:
  - **Intensity centroid** within the marker mask.
  - **Edge‑based centroid** from a Canny edge map or gradient map.
  - **Gaussian fit** to the local patch to approximate the marker as a 2D Gaussian.
- Combine them via a small learned regressor or a robust average, weighted by local contrast; this is similar in spirit to LINEEX’s transformer keypoint module which merges multiple cues. [shiva-sankaran.github](https://shiva-sankaran.github.io/assets/docs/lineex.pdf)

This improves stability across light‑on‑dark markers and anti‑aliased shapes without changing the downstream calibration code.

### Robust statistics (P1)

- In addition to mean/std and Pearson r, compute:
  - **Spearman r** to capture rank correlation when calibration is slightly non‑linear.
  - **MAD (median absolute deviation)** for robust spread.
  - **Mahalanobis distances** in the (x, y) calibrated space to flag outliers that are inconsistent with the main cluster.
- Expose these in `diagnostics['robust_stats']` so your HybridStrategy and conformal layer can down‑weight outliers and flag suspicious extractions, echoing how ChartReader and materials‑science chart extraction papers use task‑specific metrics to quantify extraction quality. [openreview](https://openreview.net/forum?id=vj8dqNrzEe)

***

## How to phase this into your codebase

- **Short term (P0)**: implement strictly local fixes that do not alter any public contract:
  - Box five‑number projection and outlier gate.
  - Scatter baseline/dual‑axis bug fixes.
  - Histogram orientation parity plus simple bin‑contiguity warnings.
- **Medium term (P1)**: introduce adaptive and ensemble logic:
  - GMM‑based bar layout detection; histogram contiguity and provenance; scatter sub‑pixel ensembles; robust stats.
- **Longer term (P2)**: train **task‑specific learned components**:
  - Bar–label association classifier and error‑bar validator, using the single‑network and object‑matching designs from Liu et al. and ChartReader. [openreview](https://openreview.net/pdf?id=8nu1AqmMhO)

All of this stays within your existing handler/extractor boundaries and dovetails with the Phase‑1 architectural/uncertainty work and your Isolation‑First experiment policy.


Modern literature gives you three clear upgrade paths: (1) replace Otsu‑only scatter refinement with subpixel keypoint localization or heatmap regression; (2) move heatmap color→value mapping into CIELAB with learned 1D spline inversion of the colorbar plus cell‑scale DBSCAN; (3) treat pie slices as keypoint‑based sectors and fit angles directly from pose keypoints so values can be normalized to sum‑to‑one. [emergentmind](https://www.emergentmind.com/topics/heatmap-regression-architectures)

***

## Scatter: subpixel point refinement beyond Otsu

Your current scatter extractor assumes dark‑on‑light markers and refines positions via a single Otsu binarization with fixed padding, which is fragile across marker styles and resolutions. SOTA localization in vision uses either explicit subpixel edge/blob fitting or deep **heatmap‑regression keypoint detectors** with subpixel decoding instead of hard thresholding. [atlantis-press](https://www.atlantis-press.com/article/25866.pdf)

### Classical subpixel localization (Gaussian/edge based)

Several classical lines of work show how to move from pixel‑accurate detections (your current YOLO + Otsu) to subpixel positions:

- **Gaussian surface fitting on Harris corners**: corners are first detected at pixel level, then a 2D Gaussian surface is fit around each corner to obtain subpixel precision; this significantly improves localization in matching tasks. [atlantis-press](https://www.atlantis-press.com/article/25866.pdf)
- **Gaussian‑based edge/spot localization**: surveys of subpixel edge detection classify methods into curve‑fitting, moment‑based, and reconstructive; fitting a 1D or 2D Gaussian to gradient magnitude along the edge yields more accurate edge and center estimates than centroid or parabola fits. [annals-csis](https://annals-csis.org/Volume_2/pliks/136.pdf)
- **Subpixel blob detectors** (Hessian‑based, gradient intersection): blob detectors estimate blob centers then refine positions by intersecting image gradients or fitting anisotropic Gaussians, achieving errors below 0.1 px even for overlapping elliptic spots. [static.aminer](https://static.aminer.org/pdf/PDF/000/323/513/fast_and_subpixel_precise_blob_detection_and_attribution.pdf)

**How this maps to your scatter markers**

- Treat each detected scatter marker as a **small blob**:
  - Use your existing detection as a coarse ROI.
  - Within the ROI, compute gradients or intensity patches and fit a 2D Gaussian (or anisotropic Gaussian if markers are elongated) to get subpixel center coordinates. [francis-press](https://francis-press.com/papers/7194)
- This is robust across different marker colors/backgrounds because it uses local intensity structure instead of a global binary threshold.

In code terms, you can replace the Otsu‑based refinement in `scatter_extractor.py` with:

- Step 1: crop a small window around each detection.
- Step 2: compute a gradient magnitude image.
- Step 3: fit a parametric Gaussian (or use moment‑based estimators) to the blob to obtain continuous \(x, y\) in pixel space. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0263224123003202)

You still pass calibrated values through your existing calibration logic; only the subpixel step changes.

### Heatmap‑regression keypoint decoding

Modern pose and keypoint detectors don’t directly regress coordinates; they **predict a heatmap channel per keypoint** and then decode continuous positions with soft‑argmax or local peak interpolation. [viblo](https://viblo.asia/p/paper-explain-whole-body-2d-human-pose-estimation-based-on-human-keypoints-distribution-constraint-and-adaptive-gaussian-factor-QyJKzwq14Me)

Key ideas:

- Ground‑truth keypoints are rendered as 2D Gaussians on a heatmap; a CNN predicts these heatmaps as likelihood fields. [viblo](https://viblo.asia/p/paper-explain-whole-body-2d-human-pose-estimation-based-on-human-keypoints-distribution-constraint-and-adaptive-gaussian-factor-QyJKzwq14Me)
- Subpixel precision is recovered by:
  - Using continuous Gaussian labels at the **true subpixel position** to reduce quantization error, and
  - Decoding with **local soft‑argmax** over a small window around the peak to estimate a continuous offset. [emergentmind](https://www.emergentmind.com/topics/heatmap-regression-architectures)
- Adaptive sigma for Gaussians (distance‑ or object‑size‑aware) further improves localization in long‑range and scale‑varying settings. [nature](https://www.nature.com/articles/s41598-025-31572-3)

**How to use this in your scatter pipeline**

You already have a detector for chart elements. For scatter points specifically, you could: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77311235/c0a42b95-945e-4507-a949-047171a75c05/paste.txt)

- Add a **scatter‑keypoint head** (or a small separate model) that predicts a heatmap for each detected point cluster, then decode subpixel positions with soft‑argmax. [emergentmind](https://www.emergentmind.com/topics/heatmap-regression-architectures)
- Alternatively, treat each detected marker as a small image and run a tiny CNN that outputs a heatmap for the marker center, decoded via a 2D soft‑argmax.

This approach is more “learned” than classical Gaussian fitting but integrates well if you later move to a fully neural ChartDETR/ChartVLM‑style shape head (see below for pies). [microsoft](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/12/WACV_2021_ChartOCR.pdf)

**Design recommendation**

- For a low‑risk upgrade, start with **Gaussian fitting inside the existing ROIs** to eliminate Otsu’s dark‑on‑light assumption. [annals-csis](https://annals-csis.org/Volume_2/pliks/136.pdf)
- If you later add a scatter‑specific model, adopt **heatmap regression + soft‑argmax** using your existing labeled scatter corpus; that gives you state‑of‑the‑art point accuracy and a natural uncertainty measure (peak sharpness) that can be folded into your conformal layer. [nature](https://www.nature.com/articles/s41598-025-31572-3)

***

## Heatmap: perceptual color mapping and DBSCAN scaling

Your heatmap handler currently sets DBSCAN `eps` as 1.5% of image height and uses a 4‑tier HSV hue fallback hard‑coded for blue→red colormaps, with no confidence measure. SOTA scalar‑field visualization and colormap‑recovery work emphasize operating in **perceptually uniform spaces (CIELAB)**, fitting continuous color→scalar mappings, and treating the colorbar as an explicit 1D function that can be inverted. [shape-of-code](https://shape-of-code.com/2015/03/04/extracting-the-original-data-from-a-heatmap-image/)

### Inverting colormaps and scalar recovery

Several practical and research works tackle “read data back from a heatmap image”:

- A blog case study on recovering Linux kernel evolution data from a heatmap describes extracting the colorbar strip, sampling RGB along it, and building a mapping from scalar values to RGB that can be inverted to map every pixel color back to a scalar. [shape-of-code](https://shape-of-code.com/2015/03/04/extracting-the-original-data-from-a-heatmap-image/)
- Q&A and practitioner discussions about digitizing heatmaps recommend:
  - Extracting the colorbar,
  - Building a palette of colors vs normalized values,
  - Then quantizing image pixels to the nearest palette color to obtain scalar values. [discourse.julialang](https://discourse.julialang.org/t/rebuilding-data-from-heatmaps/122291)
- Recent research proposes a **self‑supervised continuous colormap recovery** method that learns a colormap from a scalar field visualization by simultaneously adjusting both the colormap and data values to match the observed image, explicitly modeling a continuous color mapping instead of discrete bins. [arxiv](https://arxiv.org/pdf/2507.20632.pdf)
- “Data‑driven colormap adjustment” optimizes colormap control points for better spatial variation perception, representing colormaps in a parametric space and adjusting them in response to the underlying scalar field; this uses colormap parameterization in perceptual color spaces. [cg.tuwien.ac](https://www.cg.tuwien.ac.at/research/publications/2021/Zeng_2021/Zeng_2021-Paper.pdf)

Common threads:

- Use the colorbar (when present) to define a 1D mapping from **normalized scalar → color**.
- Work in a **perceptually uniform space** like CIELAB or CIECAM, not raw RGB or HSV, for interpolation and distance.
- Fit a **smooth function** (often spline) to map scalar to color, and invert it numerically (1D search or nearest‑neighbor in color space).

**How this maps to your color_mapping_service**

Replace hard‑coded HSV tiered mapping with:

1. **Colorbar extraction and calibration**
   - Detect the colorbar region (you likely already find it via detection/OCR in heatmaps). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77311235/c0a42b95-945e-4507-a949-047171a75c05/paste.txt)
   - Sample colorbar pixels along its length, convert to CIELAB, and associate each sample with a normalized scalar position (0 to 1), using label OCR on min/max/mid values to get the scalar range. [stackoverflow](https://stackoverflow.com/questions/3720840/how-to-reverse-a-color-map-image-to-scalar-values)
   - Fit a 1D spline or monotone curve per LAB channel as a function of scalar value; this is directly in line with continuous colormap recovery and data‑driven adjustment techniques. [arxiv](https://arxiv.org/pdf/2507.20632.pdf)

2. **Color→value inversion**
   - For each heatmap cell, compute its average CIELAB color.
   - To find the scalar value:
     - Either search for the scalar that minimizes distance between the cell’s LAB and the spline‑predicted LAB (1D optimization),
     - Or use a densely sampled lookup table over scalar ∈  and choose the nearest color in LAB space. [stackoverflow](https://stackoverflow.com/questions/3720840/how-to-reverse-a-color-map-image-to-scalar-values)
   - Use the distance in LAB between the observed cell color and the best match as a **confidence measure**, e.g. `exp(-d^2 / (2σ^2))`, and propagate it as `value_confidence` and `value_source='lab_spline'`.

3. **Fallbacks and extrapolation**
   - When cell colors lie outside the colormap path due to compression or noise, clamp to min/max scalar and record a warning, similar to how data‑recovery workflows treat values outside the colorbar’s range. [reddit](https://www.reddit.com/r/datamining/comments/cez1eb/extracting_data_from_heatmaps/)

This exactly matches the upgrades you sketched (CIELAB + spline) but ties them to published approaches on colormap inversion and recovery. [cg.tuwien.ac](https://www.cg.tuwien.ac.at/research/publications/2021/Zeng_2021/Zeng_2021-Paper.pdf)

### DBSCAN eps from cell geometry

Your current DBSCAN eps scales with image size, leading to failures on very high‑ or low‑resolution images. Heatmap digitization discussions and scalar‑field tools (ParaView, Matplotlib) treat heatmaps as **regular grids** whose cell size is defined by axis or layout, not by total image dimensions. [stackoverflow](https://stackoverflow.com/questions/60777750/digitizing-heatmap-and-map-pixels-to-values)

To make eps consistent across resolutions:

- Estimate **median cell width/height** from the initial clustering or from the grid structure (e.g., run a coarse Hough/line detection or use K‑means along one axis).
- Set `eps` as a fraction of the median cell size (e.g., 0.5× min(cell_h, cell_w)) rather than a fraction of full image height, echoing strategies used in grid‑based heatmap digitization where region boundaries are tied to grid lines rather than absolute pixels. [stackoverflow](https://stackoverflow.com/questions/49471502/how-to-digitize-extract-data-from-a-heat-map-image-using-python)
- Optionally, adapt eps differently in x and y if cells are strongly anisotropic, similar to multiscale anisotropic Gaussian methods for spot localization in scalar fields. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0263224123003202)

You can surface the derived `eps` and cell geometry in diagnostics for debugging and later learning‑based refinement.

***

## Pie: keypoints, angles, and sum‑to‑one normalization

Your pie handler ignores the 5 pose keypoints per slice provided by `Pie_pose.onnx`, instead using centroid heuristics and never enforcing that values sum to 1.0; legend alignment and angle reference mismatches add additional brittleness. There is now a clear SOTA trend toward **keypoint‑based pie slice detection and angle regression** in chart literature. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77311235/c0a42b95-945e-4507-a949-047171a75c05/paste.txt)

### Keypoint‑based pie extraction (ChartOCR, AI‑ChartParser, ChartDETR)

- **ChartOCR** introduces a deep hybrid framework where pie charts are represented by keypoints: the pie center plus arc intersection points that segment the circle into sectors. [microsoft](https://www.microsoft.com/en-us/research/publication/chartocr-data-extraction-from-charts-images-via-a-deep-hybrid-framework/)
  - They use a CornerNet‑style keypoint detector (with center pooling) to find the pie center and boundary points, then group keypoints into slices (sectors) without relying on bounding boxes. [microsoft](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/12/WACV_2021_ChartOCR.pdf)
- **AI‑ChartParser** builds on ChartOCR; it uses CNN chart‑type classification followed by a keypoint‑detection module based on CornerNet for pie charts, then reconstructs slice shapes and data from those keypoints. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70146?af=R)
- **ChartDETR** uses a DETR‑like transformer (ChartDETR) to output a **set of pie sectors directly**, with three keypoints per sector (center and two arc endpoints), grouped by “pivot queries” without extra post‑processing. [arxiv](https://arxiv.org/pdf/2308.07743.pdf)
  - This avoids the complex grouping logic needed when keypoints are detected separately and shows that pie sectors can be modeled as fundamental objects parameterized by their keypoints.

All three approaches align with your runtime assumption that pie slices come with multiple keypoints: center plus boundary points per slice. [arxiv](https://arxiv.org/pdf/2308.07743.pdf)

**How to use your existing 5‑keypoint pose outputs**

Given that `Pie_pose.onnx` yields 5 keypoints per slice (center + multiple boundary points), you can:

1. **Center estimation**
   - Average all center keypoints across slices (or fit a circle using all boundary keypoints) to get a robust global center, similar to ChartOCR’s center pooling idea. [microsoft](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/12/WACV_2021_ChartOCR.pdf)

2. **Angle computation per slice**
   - For each slice, take its boundary keypoints, compute vectors from the global center to each boundary point, and compute angles via `atan2`.
   - Sort angles and compute the **angular span** as the difference between boundary angles, exactly as in ChartDETR/ChartOCR where slices are defined by center and arc intersection points. [arxiv](https://arxiv.org/pdf/2308.07743.pdf)

3. **Grouping and ordering**
   - Use the angular positions to order slices around the circle; this naturally resolves your current mismatch between 0° at East and legend ordering from 12 o’clock, because you can:
     - Normalize angles to a fixed reference (e.g., 0° at North) and direction (clockwise vs counterclockwise).
     - Align the sorted slices with legend entries through joint color + angle proximity.

This effectively moves you from centroid heuristics to a **keypoint‑driven angle representation**, exactly what recent chart models implement. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70146?af=R)

### Sum‑to‑one normalization and data labels

Mathematically, sector angles and values in a pie chart must satisfy:

- Sum of angles = 360° (or \(2\pi\)), and
- Sum of normalized proportions = 1.0. [had2know](https://www.had2know.org/education/calculate-pie-chart-sector-angles.html)

Educational and calculator resources reiterate that the sector angle is `(component / total) * 360°`, or equivalently `(percentage * 3.6)` degrees, and that pie charts are fundamentally **part‑to‑whole** visualizations. [vedantu](https://www.vedantu.com/question-answer/in-a-pie-chart-the-angle-of-each-sector-is-class-10-maths-cbse-5fd31619c0af9500a273ce98)

Coupled with your keypoint‑derived angular spans:

- After computing raw angular spans \( \theta_i\) for each slice (in degrees or radians), normalize as:
  - \( \hat{\theta}_i = \theta_i / \sum_j \theta_j\),
  - And set values \(v_i = \hat{\theta}_i\) (if chart is percentage‑based) or scale to total from legend/labels when available.
- When OCR identifies explicit labels like “25%” or numerical values, use them to:
  - Override geometric estimates for matching slices, and
  - Infer the total (sum of labels) so geometric slices without labels can be scaled appropriately.

This corresponds exactly to your planned **data label override** and **sum‑to‑one normalization** steps and is supported by mathematical treatments of pie charts. [thirdspacelearning](https://thirdspacelearning.com/gcse-maths/statistics/pie-chart/)

### From pose keypoints to robust angles

Beyond chart‑specific work, general pose‑estimation literature offers techniques for using keypoint distributions and constraints to improve angular estimates:

- Whole‑body and 2D pose estimation use **Gaussian heatmaps with adaptive sigma** and distribution constraints to improve localization of joints, then derive angles between bones. [viblo](https://viblo.asia/p/paper-explain-whole-body-2d-human-pose-estimation-based-on-human-keypoints-distribution-constraint-and-adaptive-gaussian-factor-QyJKzwq14Me)
- Pose‑based 6D estimation for UAVs uses keypoint heatmaps with adaptive sigma distances to maintain localization accuracy across scales and distances. [nature](https://www.nature.com/articles/s41598-025-31572-3)

You can adopt similar ideas for pie charts:

- Train or fine‑tune `Pie_pose.onnx` (or its successor) with:
  - Adaptive sigma for boundary keypoints based on slice width (narrow vs wide slices).
  - Losses that enforce angular consistency (e.g., total angles ≈ 360°, non‑overlapping sectors).
- Use **heatmap‑peak sharpness** and distance between keypoints and fitted circle as confidence signals per slice, surfaced in `diagnostics['slice_confidence']`.

These techniques align your pie pose estimation with SOTA keypoint‑heatmap practice rather than static thresholds. [viblo](https://viblo.asia/p/paper-explain-whole-body-2d-human-pose-estimation-based-on-human-keypoints-distribution-constraint-and-adaptive-gaussian-factor-QyJKzwq14Me)

***

## How these address your target bottlenecks

- **Scatter sub‑pixel Otsu limitations**  
  - Replacing Otsu refinement with Gaussian‑based subpixel localization or a small heatmap‑regression head removes the dark‑on‑light assumption and offers provably more accurate and robust keypoint estimates. [emerald](https://www.emerald.com/insight/content/doi/10.1108/02602281011010790/full/html)
- **Heatmap DBSCAN scaling & HSV mapping**  
  - Setting eps from **cell geometry** instead of image size follows grid‑based heatmap digitization practice and makes clustering stable across resolutions. [stackoverflow](https://stackoverflow.com/questions/60777750/digitizing-heatmap-and-map-pixels-to-values)
  - Moving color mapping into **CIELAB with spline‑fit colormap inversion** and using the colorbar as a learned 1D mapping directly reflects modern colormap‑recovery and adjustment techniques, while giving you a natural per‑cell confidence score. [shape-of-code](https://shape-of-code.com/2015/03/04/extracting-the-original-data-from-a-heatmap-image/)
- **Pie pose keypoints and sum‑to‑one**  
  - ChartOCR, AI‑ChartParser, and ChartDETR all demonstrate that pie slices are best modeled via **center + arc keypoints**, grouping those keypoints into sectors whose angular spans define values. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70146?af=R)
  - Using your existing 5 keypoints per slice to compute angles, then normalizing and applying data‑label overrides, aligns both with this SOTA design and with the mathematical part‑to‑whole nature of pie charts. [omnicalculator](https://www.omnicalculator.com/statistics/pie-chart-angle)

These changes fit neatly into your current handler/service boundaries, and they are compatible with your conformal‑prediction and strategy‑router plans from earlier phases.


You’re right that SOTA.md isn’t yet “drop into `calibration_base.py` / `strategies/chart_to_table.py` and wire it up.” Below is a concrete, math‑level spec that fills the three gaps you called out, using formulas and procedures that are directly lifted or standard from the cited CP and Pix2Struct / DePlot literature. 

***

## 1. Non‑conformity scores for scalar values vs. bboxes/keypoints

### 1.1 Scalar regression non‑conformity (per chart element)

Standard split conformal prediction (SCP) for regression defines non‑conformity scores as absolute residuals \(s_i = |y_i - \hat \mu(x_i)|\), and uses their \((1-\alpha)\) empirical quantile as a global half‑width for prediction intervals. [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf)

For chart **scalar values** (heights, point y‑values, box summaries, heatmap cell values), you want:

- Scale‑invariance across axes with different magnitudes.
- Compatibility with your future heteroskedastic layer.

A simple and conformal‑valid choice is the **relative absolute error**, clipped away from zero:

\[
s_i^{\text{rel}} = \frac{|y_i - \hat y_i|}{\max(|y_i|, \tau)} \quad \text{with } \tau > 0
\]

- \(\hat y_i\): pipeline prediction for a specific element on the calibration set.
- \(y_i\): gold value from protocol corpus.
- \(\tau\): small floor (e.g. your minimum meaningful magnitude in chart units), prevents division by tiny values.

On the calibration set for a fixed **chart‑type + output‑family** (e.g., “bar.y”, “scatter.y”, “heatmap.value”), compute:

1. Base model predictions \(\hat y_i\).
2. Relative residuals \(s_i^{\text{rel}}\).
3. Non‑conformity quantile

\[
q_\alpha^{\text{rel}} = \text{Quantile}_{1-\alpha}\big(\{s_i^{\text{rel}}\}_{i \in \text{cal}}\big)
\]

At runtime, for a new prediction \(\hat y\), define the interval

\[
[\hat y - w(\hat y), \hat y + w(\hat y)], \quad w(\hat y) = q_\alpha^{\text{rel}} \cdot \max(|\hat y|, \tau)
\]

This is exactly split CP applied to normalized residuals, and coverage at level \(1-\alpha\) still holds marginally because a **monotone transform of the residuals** (division by \(\max(|y_i|,\tau)\)) preserves rank order. [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf)

You can implement this once per **(chart_type, value_channel)** and store:

- `cp_quantile_rel`: the scalar \(q_\alpha^{\text{rel}}\).
- `tau`: the stabilization floor used for that channel.

### 1.2 Absolute‑error variant for small‑range axes

For axes where values are always in a tight numerical range (e.g., percentages), you can instead use **pure absolute error**:

\[
s_i^{\text{abs}} = |y_i - \hat y_i|
\]

with quantile \(q_\alpha^{\text{abs}}\) and interval \([\hat y - q_\alpha^{\text{abs}}, \hat y + q_\alpha^{\text{abs}}]\). [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf)

You can support both by storing per‑channel metadata:

- `cp_mode ∈ {'relative', 'absolute'}`.
- `cp_quantile` and (for relative) `tau`.

This lets you choose absolute error specifically for normalized outputs like pie percentages where units are already standardized.

### 1.3 Bounding‑box / keypoint non‑conformity

For any future CP around **detector geometry** (e.g., bar endpoints, scatter marker bboxes, pie keypoints), you should use **IoU‑ or distance‑based** scores, as in two‑step conformal prediction for bounding boxes. [arxiv](http://arxiv.org/pdf/2403.07263v2.pdf)

For a predicted box \(\hat B_i = (\hat x_{1i}, \hat y_{1i}, \hat x_{2i}, \hat y_{2i})\) and ground‑truth box \(B_i\), define IoU:

\[
\text{IoU}(\hat B_i, B_i) = \frac{\text{area}(\hat B_i \cap B_i)}{\text{area}(\hat B_i \cup B_i)}
\]

Use non‑conformity

\[
s_i^{\text{IoU}} = 1 - \text{IoU}(\hat B_i, B_i)
\]

Then the \((1-\alpha)\) quantile \(q_\alpha^{\text{IoU}}\) gives you a tolerated IoU deficit; you can use it either:

- As a **filter** (drop detections with \(s_i^{\text{IoU}}\) above \(q_\alpha^{\text{IoU}}\)), or
- To grow/shrink predicted boxes to ensure coverage as in adaptive two‑step CP for bboxes. [eccv.ecva](https://eccv.ecva.net/virtual/2024/poster/138)

For **keypoints** like scatter marker centers or pie slice boundary points, use a **normalized Euclidean distance**:

\[
s_i^{\text{kp}} = \frac{\lVert \hat{\mathbf{p}}_i - \mathbf{p}_i \rVert_2}{\sqrt{W^2 + H^2}}
\]

- \(W, H\): width/height of the chart image.
- \(\hat{\mathbf{p}}_i, \mathbf{p}_i\): predicted and true pixel coordinates.

Again, compute \(q_\alpha^{\text{kp}}\) and treat this as the maximum expected normalized localization error. This mirrors normalization strategies in subpixel localization and camera‑calibration via heatmap regression where coordinate errors are scaled by image size or field‑of‑view. [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Wakai_Deep_Single_Image_CVPR_2024_supplemental.pdf)

You do **not** have to expose bbox/keypoint CP directly in `ExtractionResult`; you can:

- Use geometric CP just to gate detections or adjust baselines before scalar calibration.
- Keep scalar CP as the only thing surfaced in `elements[i].uncertainty`.

***

## 2. Heteroskedasticity: concrete CQR / adaptive CP spec

### 2.1 Vanilla split CP recap (i.i.d. regression)

The standard split conformal regression recipe is (notation from Tibshirani’s conformal lecture notes and JMLR work on split CP): [jmlr](https://www.jmlr.org/papers/volume25/23-1553/23-1553.pdf)

1. Split data indices into training \(I_\text{train}\), calibration \(I_\text{cal}\), and test \(I_\text{test}\).
2. Train a base predictor \(\hat \mu(x)\) on \(I_\text{train}\).
3. On calibration set, compute residuals \(r_i = |y_i - \hat \mu(x_i)|\) for \(i \in I_\text{cal}\).
4. Set

\[
q_\alpha = \text{Quantile}_{1-\alpha}\Big(\{ r_i \}_{i\in I_\text{cal}} \Big)
\]

5. For any new \(x\), output interval

\[
C_\alpha(x) = [\hat \mu(x) - q_\alpha, \hat \mu(x) + q_\alpha]
\]

This gives marginal coverage \(P\{Y \in C_\alpha(X)\} \ge 1-\alpha\) under exchangeability. [jmlr](https://www.jmlr.org/papers/volume25/23-1553/23-1553.pdf)

Your `s_i^{\text{rel}}` or `s_i^{\text{abs}}` just plug into this as the residuals.

### 2.2 Conformalized Quantile Regression (CQR) for heteroskedastic error

For **heteroskedastic** regression, conformalized quantile regression (CQR) adjusts **conditional quantile models** with split CP so that intervals adapt to regions where noise is larger. [mathworks](https://www.mathworks.com/help/stats/create-prediction-intervals-using-split-conformal-prediction.html)

Given:

- Lower and upper conditional quantile regressors \(\hat q_\ell(x)\) and \(\hat q_u(x)\), approximating true \(Q_{\alpha_\ell}(Y\mid X=x)\) and \(Q_{\alpha_u}(Y\mid X=x)\) (e.g., \(\alpha_\ell=0.1,\alpha_u=0.9\)).
- Calibration data \((x_i, y_i)\).

Define non‑conformity scores

\[
s_i^{\text{CQR}} = \max\{ \hat q_\ell(x_i) - y_i,\; y_i - \hat q_u(x_i),\; 0\}
\]

Compute the empirical quantile:

\[
q_\alpha^{\text{CQR}} = \text{Quantile}_{1-\alpha}\Big(\{ s_i^{\text{CQR}} \}_{i\in I_\text{cal}} \Big)
\]

Then for a new \(x\), your final interval is: [mathworks](https://www.mathworks.com/help/stats/create-prediction-intervals-using-split-conformal-prediction.html)

\[
C_\alpha^{\text{CQR}}(x) = [\hat q_\ell(x) - q_\alpha^{\text{CQR}},\; \hat q_u(x) + q_\alpha^{\text{CQR}}]
\]

This:

- Adapts width to heteroskedasticity (via \(\hat q_\ell,\hat q_u\)).
- Preserves \(1-\alpha\) marginal coverage via split CP.

**How to make this chart‑friendly without blowing up complexity:**

Per *(chart_type, value_family)*:

1. Train a simple **quantile regressor** (e.g., a small MLP on features like calibrated pixel height, baseline distance, log axis span) to predict lower and upper quantiles.
2. Use the above CQR formula to get `q_alpha_cqr` on your chart validation corpus.
3. At runtime, for each element, compute \(\hat q_\ell(x), \hat q_u(x)\) and output `uncertainty.interval = [q_lo - q_alpha_cqr, q_hi + q_alpha_cqr]`.

If you don’t want to maintain separate quantile models per value‑family immediately, you can still get *some* heteroskedasticity by **binning calibration scores** by a scalar feature (e.g., magnitude or axis span) and computing separate \(q_\alpha\) per bin. [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf)

### 2.3 Binning specification (cheap adaptive CP)

For a lighter‑weight alternative to full CQR, define bins over a scalar feature \(z_i\) (e.g., \(|y_i|\), or predicted value magnitude, or bar height in pixels):

1. Choose bin edges \(b_0 < b_1 < \dots < b_K\) such that each bin has roughly equal calibration mass (e.g., quantiles of \(z_i\)). [jmlr](https://www.jmlr.org/papers/volume25/23-1553/23-1553.pdf)
2. For each bin \(k\), form calibration subset
   \[
   I_k = \{ i\in I_\text{cal} : b_{k-1} \le z_i < b_k\}
   \]
3. Compute bin‑specific quantiles
   \[
   q_{\alpha,k} = \text{Quantile}_{1-\alpha}(\{ s_i \}_{i\in I_k})
   \]

At runtime:

- Compute the same feature \(z\) for a new element.
- Find its bin \(k\), then use \(q_{\alpha,k}\) to define the interval.

This is a special case of **locally adaptive CP**; under mild conditions it remains valid marginally and can improve efficiency in heteroskedastic settings. [proceedings.mlr](https://proceedings.mlr.press/v230/clarkson24a.html)

You can implement this per value‑family and keep the resulting intervals in your additive `uncertainty` dict:

```python
uncertainty = {
    "method": "cp_split_binned",
    "alpha": 0.1,
    "mode": "relative",   # or "absolute", "cqr"
    "bin_index": k,
    "interval": [lo, hi],
}
```

No changes to `ExtractionResult.elements` layout; just populate this dict when CP is enabled.

***

## 3. Calibration‑set construction and quantile computation

The CP references give a consistent recipe you can adopt directly: [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf)

### 3.1 Offline calibration algorithm for your corpus

Per *(chart_type, value_family)*:

1. **Partition dataset indices** into:
   - `train_indices`
   - `cal_indices`
   - `test_indices` (used only for off‑line validation, not for fitting CP).
2. Train the existing **extractor pipeline** (no architectural changes) on `train_indices`.
3. Run the full Standard pipeline on **calibration charts only**.
4. For each predicted `element` with known gold value:
   - Compute scalar non‑conformity \(s_i^{\text{rel}}\) or \(s_i^{\text{abs}}\) (or \(s_i^{\text{CQR}}\) if you have quantile models).
   - Record a tuple `(value_family, s_i, z_i)` where `z_i` is the binning feature if used.
5. For each `(value_family, bin)` combination:
   - Sort scores.
   - Take the \(\lceil (n_\text{cal} + 1)(1-\alpha) \rceil\)-th largest score as the quantile \(q_{\alpha,(family,bin)}\). [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf)

You can keep all `q_alpha` values plus metadata in a JSON sidecar per model version and load them in `calibration_factory.py` alongside your existing calibrators.

### 3.2 Runtime integration in `calibration_base.py`

In `CalibrationResult` or equivalent:

- Compute R² as you already do; **do not change existing semantics**.
- After calibration, for each value:

  1. If CP is enabled and calibration has enough points:
     - Compute \(w(\hat y)\) using the stored `q_alpha` (binned or CQR).
     - Attach `uncertainty` dict with interval and method metadata.
  2. Derive `calibration_quality` from R² and relative interval width:
     - `high` if R² ≥ 0.85 and average \(w(\hat y) / |\hat y|\) below a small threshold.
     - `approximate` if 0.15 ≤ R² < 0.85 and intervals moderate.
     - `uncalibrated` otherwise (including undefined R² for constant Y). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77311235/c0a42b95-945e-4507-a949-047171a75c05/paste.txt)

This leaves `FAILURE_R2 = 0.40` intact as a legacy threshold but **no longer forces a hard failure**; instead it becomes one signal in `calibration_quality`, consistent with your Critic.md roadmap. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77311235/c0a42b95-945e-4507-a949-047171a75c05/paste.txt)

***

## 4. DePlot/MatCha integration: processor, tensors, prompts, mapping

### 4.1 Pix2Struct / DePlot preprocessing and tensors

DePlot is implemented as a finetuned **Pix2StructForConditionalGeneration** and uses **Pix2StructProcessor** for both image patchification and text tokenization. [aclanthology](https://aclanthology.org/2023.findings-acl.660/)

The HuggingFace model card for `google/deplot` shows the canonical usage: [huggingface](https://huggingface.co/google/deplot)

```python
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image

processor = Pix2StructProcessor.from_pretrained("google/deplot")
model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")

image = Image.open("/path/to/chart.png").convert("RGB")
prompt = "Generate data table of the figure below:"

inputs = processor(
    images=image,
    text=prompt,
    return_tensors="pt"   # => PyTorch tensors
)

outputs = model.generate(**inputs, max_new_tokens=512)
text = processor.decode(outputs[0], skip_special_tokens=True)
```

Key points from the Pix2Struct docs and processing code: [huggingface](https://huggingface.co/docs/transformers/model_doc/pix2struct)

- `processor` internally:
  - Resizes and normalizes images according to the model config, then splits them into **patches** of size `patch_size["height"] × patch_size["width"]`, stored as flattened patches. [github](https://github.com/huggingface/transformers/blob/main/src/transformers/models/pix2struct/processing_pix2struct.py)
  - Produces a tensor of `flattened_patches` with shape:
    \[
    (\text{batch\_size}, \text{num\_patches}, \text{patch\_size}_h \cdot \text{patch\_size}_w \cdot 3)
    \]
    plus an associated `attention_mask`. [huggingface](https://huggingface.co/docs/transformers/model_doc/pix2struct)
- `return_tensors="pt"` yields PyTorch tensors. You do **not** need to manage a `[B, 3, H, W]` tensor manually; you pass PIL images and let Pix2Struct handle patch embedding. [huggingface](https://huggingface.co/google/deplot)
- Text is encoded as `input_ids` tokens appended to the visual token sequence, so you can condition on chart type and task in the prompt without changing tensor shapes. [huggingface](https://huggingface.co/docs/transformers/model_doc/pix2struct)

Your `ChartToTableStrategy.execute` therefore only needs to:

1. Convert your `np.ndarray` image to a PIL `Image`.
2. Call the shared `Pix2StructProcessor` with a **task prompt** and `return_tensors="pt"`.
3. Pass the resulting dict directly into `Pix2StructForConditionalGeneration.generate`.

No modifications to `HandlerContext` or detection stages are required.

### 4.2 Prompt specification for structured table output

DePlot’s paper defines the **plot‑to‑table task** as producing a “linearized table” consisting of rows of `(row_header, column_header, value)` triplets. It standardizes formats across chart types and trains the model end‑to‑end to emit these linearized tables as text. [aclanthology](https://aclanthology.org/2023.findings-acl.660.pdf)

In practice, your prompt can be:

- Base prompt: `"Generate the data table of the figure below:"` as used in the HF examples. [huggingface](https://huggingface.co/google/deplot)
- Optionally **conditioned on chart type**:

  - `"Generate the data table of the bar chart below:"`
  - `"Generate the data table of the scatter plot below:"`

DePlot is trained to produce outputs like (pseudo‑example):

```text
col: Year; row: Series A; val: 2018 10
col: Year; row: Series A; val: 2019 12
col: Year; row: Series B; val: 2018 5
...
```

or other linearized conventions described in the paper’s standardized task formats. [aclanthology](https://aclanthology.org/2023.findings-acl.660/)

For robust integration:

- Treat DePlot output as **plain text** and parse it with a simple state machine or regexes that recognize `row:`, `col:`, and `val:` prefixes, as shown in the paper’s examples. [aclanthology](https://aclanthology.org/2023.findings-acl.660.pdf)
- Do **not** attempt to force strict JSON via prompt at first; stick to the pretraining format described in the paper, because DePlot was trained on that linearization. [aclanthology](https://aclanthology.org/2023.findings-acl.660/)

### 4.3 Mapping DePlot output to `ExtractionResult.elements` and `calibration`

Once you have parsed DePlot’s linearized table into rows:

- Each unique `(row_header, col_header)` combination becomes a **series/group key** for your `elements` list (e.g., group name + category).
- Each associated scalar in the table becomes `value` in your existing schema. If DePlot provides the **x‑axis categories and y values**:

  - `group`: series name or legend label.
  - `category`/`x_label`: row or column header (depending on chart type mapping).
  - `value`: numeric value parsed from DePlot’s `val` token(s).

Since DePlot works in the **chart’s own numeric space** (it predicts data values, not pixel coordinates), you can:

- Set `calibration = None` or a special calibration type like `"model"` to denote that no pixel→value mapping was used.
- Use:

  ```python
  result.diagnostics["value_source"] = "chart_to_table"
  result.diagnostics["strategy_id"] = "chart_to_table"
  result.diagnostics["calibration_quality"] = "model"  # or leave unset
  ```

This keeps your `PipelineResult` schema unchanged while clearly marking that:

- Values came from DePlot rather than from `CartesianExtractionHandler` calibration.
- No R² exists for this path; any CP/UQ you attach would be **model‑level** (you can still apply CP on DePlot outputs if you evaluate them on your corpus as a separate predictor \( \hat y^{\text{DePlot}}(x)\)).

### 4.4 MatCha: pretraining and configuration

MatCha is a pretraining regime that starts from **Pix2Struct** and adds chart‑derendering and math‑reasoning tasks; architectures and processors remain those of Pix2Struct. Specifically: [aclanthology](https://aclanthology.org/2023.acl-long.714.pdf)

- MatCha pretrains on:
  - Plot deconstruction (predict tables and plotting code).
  - Numerical reasoning datasets (MATH, DROP) to improve math reasoning. [arxiv](https://arxiv.org/abs/2212.09662)
- DePlot is then finetuned *from* MatCha on the plot‑to‑table task, reusing Pix2Struct’s processor/model stack. [arxiv](https://arxiv.org/abs/2212.09662)

Implication:

- You do **not** need MatCha‑specific processors; using `Pix2StructProcessor` with the DePlot weights already embeds the MatCha pretraining. [aclanthology](https://aclanthology.org/2023.acl-long.714.pdf)
- Any additional chart‑type conditioning can be injected at the **text level** in the prompt.

From an implementation standpoint, your `ChartToTableStrategy` can be fully specified as:

```python
class ChartToTableStrategy(PipelineStrategy):
    STRATEGY_ID = "chart_to_table"

    def __init__(self, deplot_model, deplot_processor, alpha: float = 0.1):
        self.model = deplot_model       # Pix2StructForConditionalGeneration
        self.processor = deplot_processor
        self.alpha = alpha              # for any optional CP layer on deplot outputs

    def execute(self, image: np.ndarray, chart_type: str,
                classification_confidence: float,
                services: StrategyServices) -> ExtractionResult:

        pil_image = Image.fromarray(image).convert("RGB")

        prompt = f"Generate the data table of the {chart_type} below:"

        inputs = self.processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt"
        )
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        text = self.processor.decode(outputs[0], skip_special_tokens=True)

        table_rows = parse_deplot_table(text)  # your parser for linearized table
        elements = map_table_to_elements(table_rows)  # produce ExtractionResult.elements

        # Optionally apply scalar CP on DePlot’s values if you have calibrated q_alpha
        # cp_uncertainty = attach_cp_uncertainty(elements, self.alpha, ...)

        return ExtractionResult(
            elements=elements,
            calibration=None,  # or a special marker
            baselines=[],
            errors=[],
            warnings=[],
            diagnostics={
                "strategy_id": "chart_to_table",
                "value_source": "chart_to_table",
                # "uncertainty_method": "cp_split" if you conformalize DePlot outputs
            },
        )
```

This uses the exact HF processor/model contract, the prompt pattern recommended by the DePlot model card, and DePlot’s published linearized table semantics. [aclanthology](https://aclanthology.org/2023.findings-acl.660.pdf)

***

## 5. How this keeps contracts and Isolation‑First gates intact

- **`ExtractionResult.elements`** remains a flat list of value records; you only add an optional `uncertainty` dict per element and a few diagnostics fields, all already reserved as non‑breaking additions in Critic.md. [ppl-ai-file-upload.s3.amazonaws]
- **`calibration/calibration_base.py`** continues to compute R² and existing statistics; you layer CP *on top* of that by:

  - Computing non‑conformity scores on your frozen validation corpus.
  - Storing global or binned quantiles \(q_\alpha\).
  - Attaching intervals and `calibration_quality` instead of hard‑failing below `FAILURE_R2`.

- **`StrategyRouter` and `ChartToTableStrategy`** use DePlot/MatCha exactly as specified in the DePlot and Pix2Struct documentation (processor API, tensor shapes, generation call, decoding), so there is no API guesswork at the tensor level. [arxiv](https://arxiv.org/abs/2212.09662)
- All new behavior is **feature‑flagged** (CP enabled flag, `pipeline_mode='chart_to_table'`), so you can meet the Isolation‑First policy: unchanged defaults, A/B via protocol harness, and explicit rollback paths. [ppl-ai-file-upload.s3.amazonaws]


You’re right: the earlier write‑up was architecture‑correct but implementation‑useless. Below is a concrete, math‑level spec for the three mandated areas that you can drop into `bar_associator.py`, `scatter_extractor.py`, and related helpers without touching `ExtractionResult` or protocol schemas. [atlantis-press](https://www.atlantis-press.com/article/25866.pdf)

Where the cited chart papers (Liu et al. “Data Extraction from Charts via Single Deep Neural Network”, ChartReader) **do not** actually spell out metric‑learning or subpixel math, I call that out explicitly and then give you a self‑contained design that is consistent with your runtime and with standard metric‑learning / subpixel practice. [cs.odu](https://www.cs.odu.edu/~jwu/downloads/pubs/rane-2021-iri/rane-2021-iri.pdf)

***

## 1. Metric learning for bar–label association

### 1.1 What the chart papers actually do

- **Liu et al. 2019 (Data Extraction from Charts via Single Deep Neural Network)** predict bars and text with a single CNN and then use a **classification branch** to associate bars to text regions, not a contrastive embedding. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/1906.11906)
- **ChartReader** uses a deep classifier for chart type and then **heuristic, geometry‑driven matching** for axis ticks and legends, not metric learning. [cs.odu](https://www.cs.odu.edu/~jwu/downloads/pubs/rane-2021-iri/rane-2021-iri.pdf)

So neither paper gives you a ready‑made triplet/InfoNCE embedding. You’ll need to design your own, but you can reuse the same *kind* of geometric and appearance features they use for their heuristics (Δx, Δy, alignment, overlaps, colors, confidences). [arxiv](https://arxiv.org/pdf/1906.11906.pdf)

### 1.2 Feature vector specification

For each candidate pair (bar \(b\), label \(t\)) define a **fixed‑dimensional feature vector** \( \mathbf{f}(b,t) \in \mathbb{R}^{d} \) with \(d=16\). All coordinates are resolution‑normalized so the model generalizes across image sizes.

Let:

- Image width/height \(W, H\).
- Bar bbox \(b = (x_b, y_b, w_b, h_b)\) in pixels (top‑left + width/height).
- Label bbox \(t = (x_t, y_t, w_t, h_t)\).
- Bar center \(c_b = (x_b + w_b/2, y_b + h_b/2)\).
- Label center \(c_t = (x_t + w_t/2, y_t + h_t/2)\).
- Detector confidences \(p_b, p_t\).
- Per‑region mean colors in CIELAB: \(\mathbf{c}_b, \mathbf{c}_t \in \mathbb{R}^3\).

Define:

1. Normalized offsets  
   \[
   \Delta x = \frac{x_t - x_b}{W},\quad
   \Delta y = \frac{y_t - y_b}{H}
   \]
   \[
   \Delta x_c = \frac{c_t^x - c_b^x}{W},\quad
   \Delta y_c = \frac{c_t^y - c_b^y}{H}
   \]

2. Normalized distance and angle  
   \[
   d_{ct} = \sqrt{\Delta x_c^2 + \Delta y_c^2},\quad
   \phi_{ct} = \operatorname{atan2}(\Delta y_c,\, \Delta x_c)
   \]

3. Size and overlap ratios  
   \[
   r_w = \frac{w_t}{w_b + \epsilon},\quad
   r_h = \frac{h_t}{h_b + \epsilon}
   \]
   Horizontal and vertical overlap (IoU in 1D):
   \[
   o_x = \frac{\text{len}([x_b, x_b+w_b] \cap [x_t, x_t+w_t])}{\min(w_b, w_t)},\quad
   o_y = \frac{\text{len}([y_b, y_b+h_b] \cap [y_t, y_t+h_t])}{\min(h_b, h_t)}
   \]

4. Appearance and confidence  
   - LAB color difference:
     \[
     d_{\text{lab}} = \lVert \mathbf{c}_b - \mathbf{c}_t \rVert_2
     \]
   - Detector confidences (already \(\in [0,1]\)): \(p_b, p_t\).

Stack these into:

\[
\mathbf{f}(b,t) = [\Delta x, \Delta y, \Delta x_c, \Delta y_c, d_{ct}, \phi_{ct}, r_w, r_h, o_x, o_y, d_{\text{lab}}, p_b, p_t, 1_{\text{same\_cluster}}, 1_{\text{left\_of}}, 1_{\text{above}}]
\]

- `1_same_cluster`: optional bit from your existing meta‑clustering (e.g., bar and text assigned to same x‑cluster).
- `1_left_of`, `1_above`: bits encoding relative orientation.

This is a 16‑dimensional vector per pair and can be computed from your existing detection and clustering outputs. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77311235/c0a42b95-945e-4507-a949-047171a75c05/paste.txt)

### 1.3 Embedding + contrastive loss

Use a **Siamese MLP** \(g_\theta: \mathbb{R}^{d} \rightarrow \mathbb{R}^{D}\) (embedding) with \(D=32\):

- Two hidden layers, e.g. \(d \rightarrow 64 \rightarrow 32\), ReLU activations.
- L2‑normalize the output: \(\mathbf{z}(b,t) = \frac{g_\theta(\mathbf{f}(b,t))}{\|g_\theta(\mathbf{f}(b,t))\|_2}\).

Define cosine similarity:

\[
s(b,t) = \mathbf{z}(b,t)^\top \mathbf{w}
\]

where \(\mathbf{w} \in \mathbb{R}^{D}\) is a learned weight vector (or simply use dot product between embeddings if you want pairwise similarity).

**Loss: InfoNCE over pairs per bar (recommended)**

For each bar \(b\) in a training batch:

- \(t^+\): its true label (according to gold).
- \(\{t_j^-\}\): all other labels on the same chart (or mini‑batch).

Define scores:

\[
s^+ = s(b, t^+),\quad s_j^- = s(b, t_j^-)
\]

The InfoNCE loss per bar is: [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf)

\[
L_b = -\log \frac{\exp(s^+ / \tau)}{\exp(s^+ / \tau) + \sum_j \exp(s_j^- / \tau)}
\]

with temperature \(\tau \in (0,1]\) (e.g., \(\tau = 0.1\)).  

Total loss over a mini‑batch \(\mathcal{B}\):

\[
L = \frac{1}{|\mathcal{B}|} \sum_{b \in \mathcal{B}} L_b
\]

This encourages the positive pair to have higher similarity than all negatives and gives you a **learned similarity score** directly usable in your associator.

If you prefer **triplet loss**, use:

\[
L_b^{\text{triplet}} = \max\{0,\, d(b,t^+) - d(b,t^-) + m\}
\]

with distance \(d = -s\) or Euclidean distance in embedding space and margin \(m\) (e.g., 0.2–0.5). [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf)

### 1.4 Training and mining strategy

- **Positives**: for each (bar, label) pair linked by your current associator on gold data, use it as \(t^+\). Optionally filter by confidence.
- **Negatives**: all other labels on the same chart that are not gold‑linked to that bar. To limit complexity, restrict to labels within some max normalized distance \(d_{ct} < d_{\max}\) (e.g., 0.5), which both ChartReader and Liu et al. implicitly assume by using local heuristics. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/1906.11906)
- Use mini‑batches of bars from multiple charts; training is standard supervised metric learning.

### 1.5 Inference and integration without schema changes

In `bar_associator.py`:

1. For each bar \(b\) and each candidate label \(t\) in a spatial window, compute \(\mathbf{f}(b,t)\) and \(s(b,t)\).
2. Construct a cost matrix \(C\) where \(C_{ij} = -s(b_i, t_j)\) for Hungarian matching.
3. Solve for 1‑to‑1 assignments **per x‑cluster** or chart, exactly as your current 4‑tier heuristic does, but now driven by learned scores instead of thresholds.
4. Post‑filter extremely low scores (e.g., no match if \(s < s_{\min}\)) to preserve your existing “no label” semantics.

All existing grouping (`group` IDs) and `ExtractionResult.elements` logic can remain unchanged; you simply swap “score this (bar,label)” from heuristics to `s(b,t)`.

***

## 2. Sub‑pixel Gaussian refinement for scatter

### 2.1 Parametric 2D Gaussian

The Harris sub‑pixel paper you cited fits a **Gaussian surface** to the intensity neighborhood around each detected corner to obtain sub‑pixel coordinates. For scatter markers, the same idea applies: treat each marker as a bright/dark blob and fit: [atlantis-press](https://www.atlantis-press.com/article/25866.pdf)

Axis‑aligned anisotropic Gaussian plus offset:

\[
G(x,y;\theta) = A \exp\left( -\frac{(x-\mu_x)^2}{2\sigma_x^2} - \frac{(y-\mu_y)^2}{2\sigma_y^2} \right) + C
\]

Parameters:

\[
\theta = (A, \mu_x, \mu_y, \sigma_x, \sigma_y, C)
\]

- \(A\): amplitude.
- \((\mu_x,\mu_y)\): **sub‑pixel center** (what you need).
- \(\sigma_x,\sigma_y\): spreads.
- \(C\): background offset.

You can omit rotation; axis‑aligned ellipses are sufficient for typical circular/square markers and align with the “Gaussian surface” usage in subpixel Harris literature. [atlantis-press](https://www.atlantis-press.com/article/25866.pdf)

### 2.2 Least‑squares objective and LM solver

Given a grayscale patch \(I(x,y)\) of size \(K\times K\) around a coarse detection \((x_0, y_0)\), define residuals:

\[
r_{ij}(\theta) = I(x_i, y_j) - G(x_i, y_j; \theta)
\]

Objective:

\[
J(\theta) = \sum_{i,j} r_{ij}(\theta)^2
\]

Minimize \(J(\theta)\) with **Levenberg–Marquardt** (LM):

Update rule:

\[
\theta^{(k+1)} = \theta^{(k)} - (J'^\top J' + \lambda I)^{-1} J'^\top \mathbf{r}
\]

- \(J'\): Jacobian of residuals wrt \(\theta\) evaluated at \(\theta^{(k)}\).
- \(\mathbf{r}\): stacked residuals \(r_{ij}(\theta^{(k)})\).
- \(\lambda\): damping parameter (e.g., start at \(10^{-3}\), adjust per standard LM schedule).

Stop when:

- Relative change in \(J\) is below a tolerance (e.g. \(|J^{(k+1)} - J^{(k)}|/J^{(k)} < 10^{-6}\)), or
- Parameter update \(\lVert \theta^{(k+1)} - \theta^{(k)} \rVert\) is below a small threshold, or
- Max iterations (e.g., 20) reached.

This is exactly the “Gaussian surface fitting” the Harris paper uses for corners; they show that the peak of the Gaussian (its extreme value) gives subpixel localization. [atlantis-press](https://www.atlantis-press.com/article/25866.pdf)

### 2.3 Initialization and robustness

Good initial \(\theta^{(0)}\):

- \(\mu_x^{(0)}, \mu_y^{(0)}\): **intensity centroid** of the patch:
  \[
  \mu_x^{(0)} = \frac{\sum I(x,y) x}{\sum I(x,y)},\quad
  \mu_y^{(0)} = \frac{\sum I(x,y) y}{\sum I(x,y)}
  \]
- \(\sigma_x^{(0)}, \sigma_y^{(0)}\): estimated from second moments (standard deviations) in x and y.
- \(A^{(0)} = \max I - \min I\), \(C^{(0)} = \min I\).

You can reject or fall back to centroid if:

- LM does not converge within max iterations, or
- Fitted \(\sigma_x,\sigma_y\) leave a plausible range (e.g. outside [0.3, 3.0] pixels).

### 2.4 Integration with your calibration and dual‑axis logic

For each scatter detection:

1. Replace the current Otsu‑refined centroid with \((\hat \mu_x, \hat \mu_y)\) from Gaussian fitting.
2. Pass these continuous pixel coordinates into your existing calibration code; you do **not** change the calibration formulas, only the input pixel locations.
3. Because Gaussian fitting generally **reduces localization error below 0.1 px** in classical settings, it should improve downstream R²; but you keep your R² and CP gating unchanged. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0263224123003202)
4. Dual‑axis safety net stays the same logically (except for your already‑planned bugfix of not aliasing calibrations); you’re just giving it better pixel coordinates.

If you later want the **heatmap‑regression alternative**, you can train a small keypoint head to output a Gaussian heatmap \(H(x,y)\) and decode centers as:

\[
\hat \mu_x = \sum_{x,y} x \cdot \frac{\exp(H(x,y)/\tau)}{\sum_{x',y'} \exp(H(x',y')/\tau)},\quad
\hat \mu_y = \sum_{x,y} y \cdot \frac{\exp(H(x,y)/\tau)}{\sum_{x',y'} \exp(H(x',y')/\tau)}
\]

(soft‑argmax with temperature \(\tau\)) trained with MSE to a ground‑truth Gaussian centered at true coordinates, with \(\sigma\) proportional to expected marker size. That’s orthogonal and can be benchmarked against the Gaussian‑fitting path inside `scatter_extractor.py`. [emergentmind](https://www.emergentmind.com/topics/heatmap-regression-architectures)

***

## 3. 1D GMM layout detection for grouped vs. simple bars

Here you want a **resolution‑normalized, probabilistic replacement** for `max_spacing > 2.5 × min_spacing` using a tiny 1D GMM over inter‑bar gaps.

### 3.1 Data: normalized gaps

Given all bar centers sorted along the x‑axis: \(c_1^x < c_2^x < \dots < c_n^x\).

Define raw gaps:

\[
d_i = c_{i+1}^x - c_i^x,\quad i=1,\dots,n-1
\]

Normalize them to be scale‑free:

\[
\tilde d_i = \frac{d_i}{\bar{w}_b}
\]

- \(\bar{w}_b\): median bar width in pixels (or median of all bar widths).
- This normalization makes gaps comparable across resolutions and bar sizes.

### 3.2 GMM model and EM updates

Fit a 1D Gaussian Mixture with \(K \in \{1,2\}\) components to \(\tilde d_i\):

\[
p(\tilde d) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\tilde d \mid \mu_k, \sigma_k^2)
\]

Parameters:

- Mixture weights \(\pi_k\) (\(\sum \pi_k = 1\)).
- Means \(\mu_k\).
- Variances \(\sigma_k^2\) (diagonal covariance in 1D).

**Initialization**

- For \(K=1\): \(\mu_1 = \text{mean}(\tilde d_i)\), \(\sigma_1^2 = \text{var}(\tilde d_i)\), \(\pi_1=1\).
- For \(K=2\): K‑means on \(\tilde d_i\) into 2 clusters; initialize \(\mu_k\) to cluster means, \(\sigma_k^2\) to cluster variances, \(\pi_k\) to cluster proportions.

**EM steps** (for each K separately):

E‑step: responsibilities

\[
\gamma_{ik} = \frac{\pi_k \mathcal{N}(\tilde d_i \mid \mu_k, \sigma_k^2)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(\tilde d_i \mid \mu_j, \sigma_j^2)}
\]

M‑step:

\[
N_k = \sum_i \gamma_{ik},\quad
\pi_k = \frac{N_k}{N},\quad
\mu_k = \frac{1}{N_k} \sum_i \gamma_{ik} \tilde d_i
\]

\[
\sigma_k^2 = \frac{1}{N_k} \sum_i \gamma_{ik} (\tilde d_i - \mu_k)^2
\]

Iterate until:

- Change in log‑likelihood \(|\ell^{(t+1)} - \ell^{(t)}| < 10^{-6}\), or
- Max iterations, e.g. 50.

Log‑likelihood:

\[
\ell_K = \sum_{i=1}^{N} \log \left( \sum_{k=1}^K \pi_k \mathcal{N}(\tilde d_i \mid \mu_k, \sigma_k^2) \right)
\]

### 3.3 BIC model selection and grouping rule

BIC for model with \(K\) components:

\[
\text{BIC}(K) = -2 \ell_K + p_K \log N
\]

Parameter count \(p_K\) for 1D GMM:

- Weights: \(K-1\) (last weight determined by others).
- Means: \(K\).
- Variances: \(K\).

So \(p_K = (K-1) + K + K = 3K - 1\).

Compare:

- Compute \(\text{BIC}(1)\) and \(\text{BIC}(2)\).
- If \(\text{BIC}(2) + \delta < \text{BIC}(1)\) (with small safety margin \(\delta\), e.g. 2–5 to avoid overfitting tiny samples), accept \(K=2\) as “grouped layout”.

Assign each gap to a component:

\[
k(i) = \arg\max_k \gamma_{ik}
\]

Let:

- Small‑gap component \(k_s\): the one with smaller \(\mu_k\).
- Large‑gap component \(k_\ell\): the other.

Treat gaps assigned to \(k_\ell\) as **group separators**:

- Whenever \(\tilde d_i\) belongs to \(k_\ell\), start a new bar group after bar \(i\).
- That gives you group IDs consistent with your existing `group_id` semantics, but now learned from a 1D probabilistic mixture instead of `max_spacing > 2.5×min_spacing`.

If \(K=1\), or if \(N\) is too small (e.g. fewer than 3 gaps), fall back to **single‑group** or your legacy heuristic; this prevents increased hard‑failure in edge cases.

### 3.4 Histogram re‑use

Exactly the same machinery can be reused for histograms:

- Use bar centers along the histogram axis (x for vertical bars; y for horizontal).
- Normalize by median bin width instead of bar width.
- GMM/BIC tells you whether there are multiple gap scales (e.g., missing bins).

This fits cleanly into `histogram_extractor.py` without touching output structures.

***

These specifications give you:

- A **16‑dimensional, explicitly defined feature vector** and InfoNCE / triplet loss for bar–label metric learning, fully pluggable into your existing associator + Hungarian without schema changes. [arxiv](https://arxiv.org/pdf/1906.11906.pdf)
- A **concrete 2D Gaussian model**, LM objective, initialization, and convergence criteria for sub‑pixel scatter refinement, with clearly defined integration points back into your calibration and dual‑axis code. [nature](https://www.nature.com/articles/s41598-018-19379-x)
- A **complete 1D GMM EM + BIC formulation** for inter‑bar gaps, including normalization, parameter counts, and grouping rules, to replace `2.5×` heuristics while preserving behavior under your Isolation‑First gates.



## Heatmap: CIELAB B‑spline colormap and inversion

### 1.1 Scalar parameterization and sampling

From your existing heatmap handler:

- Detect the colorbar and sample it along its long axis at \(N\) points (e.g. \(N=64\) or \(128\)). [shape-of-code](https://shape-of-code.com/2015/03/04/extracting-the-original-data-from-a-heatmap-image/)
- Let \(s_k \in [0,1]\) be the normalized position of sample \(k\) along the bar (0 at low end, 1 at high end, monotone in display order).
- Convert each sampled RGB color to CIELAB: \(\mathbf{y}_k = (L_k^*, a_k^*, b_k^*)\). [cg.tuwien.ac](https://www.cg.tuwien.ac.at/research/publications/2021/Zeng_2021/Zeng_2021-Paper.pdf)

So you have tuples \((s_k, \mathbf{y}_k)\) for \(k=1,\dots,N\).

### 1.2 Cubic B‑spline per LAB channel

Use a **cubic B‑spline** (degree \(p=3\)) in the scalar parameter \(s\) for each LAB channel independently. [arxiv](https://arxiv.org/pdf/2507.20632.pdf)

Choose \(M\) control points with parameter values \(u_i \in [0,1]\), \(i=0,\dots,M-1\) (e.g. \(M \ll N\), say 8–16). A simple choice is uniform spacing:

\[
u_i = \frac{i}{M-1},\quad i=0,\dots,M-1
\]

Define a **clamped, uniform knot vector** \(t = \{t_j\}_{j=0}^{M+p+1}\):

- First and last knots repeated \(p+1 = 4\) times for clamping:
  \[
  t_0 = t_1 = t_2 = t_3 = 0,\quad
  t_{M} = t_{M+1} = t_{M+2} = t_{M+3} = 1
  \]
- Interior knots uniformly spaced between 0 and 1:
  \[
  t_{4}, \dots, t_{M-1} \text{ linearly spaced in } (0,1)
  \]

Cubic B‑spline basis functions \(B_i^{(3)}(s)\) are defined recursively as usual. [arxiv](https://arxiv.org/pdf/2507.20632.pdf)

For each channel \(c \in \{L,a,b\}\), approximate the colorbar as:

\[
f_c(s) = \sum_{i=0}^{M-1} c_{i}^{(c)}\, B_i^{(3)}(s)
\]

with unknown coefficients \(c_i^{(c)}\).

Fit coefficients by least squares to the sampled data:

For each channel \(c\), solve:

\[
\min_{\mathbf{c}^{(c)} \in \mathbb{R}^{M}} \sum_{k=1}^{N} \left(f_c(s_k) - y_{k}^{(c)}\right)^2
\]

This is a linear system:

- Design matrix \(A \in \mathbb{R}^{N \times M}\), \(A_{ki} = B_i^{(3)}(s_k)\).
- Solve \(A \mathbf{c}^{(c)} \approx \mathbf{y}^{(c)}\) via normal equations or QR/SVD.

Result: a smooth mapping

\[
f: [0,1] \rightarrow \mathbb{R}^3, \quad f(s) = \left(f_L(s), f_a(s), f_b(s)\right)
\]

that approximates the colorbar path in CIELAB space. [cg.tuwien.ac](https://www.cg.tuwien.ac.at/research/publications/2021/Zeng_2021/Zeng_2021-Paper.pdf)

### 1.3 Monotonicity and ordering

Monotonicity in **perceptual scalar value** is largely guaranteed by:

- Taking samples in the **visual order along the bar** (you already know which end is low/high from text labels).
- Using a clamped cubic spline in that parameter; it preserves point order along the curve in CIELAB. [cg.tuwien.ac](https://www.cg.tuwien.ac.at/research/publications/2021/Zeng_2021/Zeng_2021-Paper.pdf)

If you want stricter monotone behavior in scalar order (e.g., to avoid local back‑tracking in brightness), you can:

- Apply **1D isotonic regression** on L\(^*\) vs \(s\) before fitting, then fit splines to the isotonic‑smoothed L\(^*\) values. [cg.tuwien.ac](https://www.cg.tuwien.ac.at/research/publications/2021/Zeng_2021/Zeng_2021-Paper.pdf)

This is optional and doesn’t change the basic spline math.

### 1.4 Inverting CIELAB → scalar \(s\)

Given a heatmap cell’s average CIELAB color \(\mathbf{y}_\text{obs} = (L^*, a^*, b^*)\), define a **distance function**:

\[
D(s) = \left\| f(s) - \mathbf{y}_\text{obs} \right\|_2
\]

You want:

\[
s^* = \arg\min_{s \in [0,1]} D(s)
\]

This is a smooth scalar function on, so 1D numerical optimization is cheap and stable. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77311235/c0a42b95-945e-4507-a949-047171a75c05/paste.txt)

Algorithm:

1. **Initialization**:
   - Find the nearest sampled colorbar sample: \(k^* = \arg\min_k \|\mathbf{y}_k - \mathbf{y}_\text{obs}\|_2\).
   - Set initial bracket \([a,b]\) around \(s_{k^*}\), e.g.  
     \(a = \max(0, s_{k^*} - h)\), \(b = \min(1, s_{k^*} + h)\) with \(h = 1/(M-1)\).

2. **1D search (Brent’s method)**:
   - Use Brent’s method (golden‑section search + parabolic interpolation) on \(D(s)\) in \([a,b]\).
   - Stop when interval width \(b-a < \epsilon_s\) with \(\epsilon_s \approx 10^{-3}\), which gives scalar precision better than 0.01 over. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77311235/c0a42b95-945e-4507-a949-047171a75c05/paste.txt)

3. Set \(s^*\) to the minimizer returned by Brent.

Because the colorbar is 1D in CIELAB and parameterized monotonically, this argmin is unique in practice. [arxiv](https://arxiv.org/pdf/2507.20632.pdf)

### 1.5 Scalar value and confidence

Let the heatmap’s scalar range be \([v_{\min}, v_{\max}]\) from your OCR of colorbar tick labels. [blaylockbk.github](https://blaylockbk.github.io/Carpenter_Workshop/matplotlib-colorbar.html)

- Map \(s^*\) to scalar:
  \[
  v = v_{\min} + s^* (v_{\max} - v_{\min})
  \]

Define **distance‑based confidence**:

- Minimal color mismatch:
  \[
  d_{\min} = D(s^*) = \left\| f(s^*) - \mathbf{y}_\text{obs} \right\|_2
  \]
- Convert to \([0,1]\) confidence via:
  \[
  \text{conf} = \exp\left(-\frac{d_{\min}^2}{2\sigma_{\text{lab}}^2}\right)
  \]
  where \(\sigma_{\text{lab}}\) is a scale hyperparameter (e.g., \(\sigma_{\text{lab}} \approx 5\) in CIELAB units). [cg.tuwien.ac](https://www.cg.tuwien.ac.at/research/publications/2021/Zeng_2021/Zeng_2021-Paper.pdf)

Surface this as:

```python
value_source = "lab_spline"
value = v
uncertainty = {
    "method": "lab_spline",
    "lab_distance": d_min,
    "confidence": conf,
}
```

No schema changes are required; `GridChartHandler` still emits the same `elements`, but each cell now has a principled `uncertainty` and `value_source`.

***

## Pie: keypoint geometry, center fit, and sum‑to‑one

### 2.1 Robust global center from slice keypoints

From ChartOCR / AI‑ChartParser / ChartDETR:

- Pie slices are modeled as **sectors of a circle** defined by a common center and slice boundary keypoints. [microsoft](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/12/WACV_2021_ChartOCR.pdf)
- They use keypoints for arc intersections and a shared center, often with RANSAC to reject outlier keypoints. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70146?af=R)

Your `Pie_pose.onnx` gives **5 keypoints per slice** (1 center‑like and 4 boundary points, or similar). To compute a robust global center \( \mathbf{c} = (c_x, c_y)\) and radius \(r\), use **least‑squares circle fitting** with RANSAC: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77311235/c0a42b95-945e-4507-a949-047171a75c05/paste.txt)

1. Collect all non‑center keypoints \(\mathbf{p}_k = (x_k, y_k)\) from all slices (boundary points).  
2. RANSAC loop:
   - Randomly sample 3 distinct points \(\mathbf{p}_a, \mathbf{p}_b, \mathbf{p}_c\) and compute the unique circle through them (if non‑collinear).
   - Circle through three points (Kåsa’s method): solve the linear system
     \[
     x^2 + y^2 + D x + E y + F = 0
     \]
     for unknowns \(D,E,F\), then
     \[
     c_x = -\frac{D}{2},\quad c_y = -\frac{E}{2},\quad r = \sqrt{c_x^2 + c_y^2 - F}
     \]
   - For this candidate circle, compute residuals:
     \[
     \epsilon_k = \big|\| \mathbf{p}_k - \mathbf{c} \|_2 - r\big|
     \]
   - Count inliers with \(\epsilon_k < \epsilon_r\) (e.g. \(\epsilon_r = 2\) px).
   - Keep the candidate with the largest inlier set.

3. Refine with **nonlinear least squares** on inliers:

   Minimize:
   \[
   J(\mathbf{c}, r) = \sum_{k \in \text{inliers}} \left( \| \mathbf{p}_k - \mathbf{c} \|_2 - r \right)^2
   \]

   This can be solved with Gauss–Newton or LM in 3 parameters \((c_x, c_y, r)\). [arxiv](https://arxiv.org/pdf/2308.07743.pdf)

This gives you a robust **global center and radius** even with noisy keypoints.

### 2.2 Slice angles from boundary keypoints

For each slice \(i\):

1. Identify its **boundary keypoints** \(\{\mathbf{p}_{i1},\dots,\mathbf{p}_{iM}\}\) (all keypoints from that slice except the center‑like one if present).
2. Compute angles of each boundary point relative to global center:

   \[
   \theta_{ij} = \operatorname{atan2}(y_{ij} - c_y,\ x_{ij} - c_x)
   \]
   Normalize to \([0, 2\pi)\):

   \[
   \theta_{ij}' = \begin{cases}
     \theta_{ij} + 2\pi & \text{if } \theta_{ij} < 0 \\
     \theta_{ij} & \text{otherwise}
   \end{cases}
   \]

3. Sort \(\theta_{ij}'\) for slice \(i\): \(\theta_{i(1)}' \le \dots \le \theta_{i(M)}'\).

For a slice that spans a contiguous arc, you want the start and end angle. A simple, robust rule given 4 boundary points is:

- Treat the **smallest and largest** angles as endpoints:
  \[
  \theta_{i,\text{start}} = \theta_{i(1)}',\quad
  \theta_{i,\text{end}} = \theta_{i(M)}'
  \]
- Angular span:
  \[
  \Delta \theta_i = \theta_{i,\text{end}} - \theta_{i,\text{start}}
  \]

This assumes each slice’s keypoints are restricted to that slice’s arc, as in ChartDETR’s 3‑keypoint representation. If you suspect wrap‑around (slice crossing \(2\pi \rightarrow 0\)), you can instead: [arxiv](https://arxiv.org/pdf/2308.07743.pdf)

- Compute all pairwise angular differences and choose the pair giving the **maximal contiguous span** under circular wrap. [arxiv](https://arxiv.org/pdf/2308.07743.pdf)

The result is a positive span \(\Delta \theta_i \in (0, 2\pi)\) per slice.

### 2.3 Normalization and sum‑to‑one with data labels

Let \(\Delta \theta_i\) be the geometric span for slice \(i\). Compute:

\[
T = \sum_i \Delta \theta_i
\]

The geometric fraction of the pie:

\[
g_i = \frac{\Delta \theta_i}{T}
\]

Now integrate **OCR‑derived labels**:

For each slice, parse a possible value label:

- If it’s a percentage string (e.g., “25%”), convert to fraction \(\lambda_i = \text{percent}/100\).
- If it’s an absolute value, you can either:
  - Defer absolute consistency checking and still normalize to fractions, or
  - Use only percentages for now.

Partition slices:

- Labeled set \(L = \{i : \lambda_i \text{ is defined}\}\)
- Unlabeled set \(U = \{i : \lambda_i \text{ undefined}\}\)

**Case A: label fractions are self‑consistent (\(0 < \sum_{i\in L} \lambda_i \le 1\))**

1. Let:
   \[
   L_{\text{sum}} = \sum_{i\in L} \lambda_i,\quad
   U_{\text{share}} = 1 - L_{\text{sum}}
   \]

2. Let geometric fractions for unlabeled slices sum:
   \[
   G_U = \sum_{j\in U} g_j
   \]

3. Define final values:

   - For labeled slices \(i \in L\):
     \[
     v_i = \lambda_i
     \]
   - For unlabeled slices \(j \in U\):
     \[
     v_j = U_{\text{share}} \cdot \frac{g_j}{G_U}
     \]

By construction:

\[
\sum_i v_i = L_{\text{sum}} + U_{\text{share}} = 1
\]

This is exactly the “labels override geometry where present; geometry shares the leftover proportionally” behavior you wanted. [had2know](https://www.had2know.org/education/calculate-pie-chart-sector-angles.html)

**Case B: labels overshoot (\(\sum_{i\in L} \lambda_i > 1\)) or are inconsistent**

Fallback strategies:

- Option 1 (strict geometry): ignore label magnitudes and use **pure geometry**:
  \[
  v_i = g_i,\ \forall i
  \]
- Option 2 (normalize labels only if all slices labeled): if \(U=\varnothing\),
  \[
  v_i = \frac{\lambda_i}{\sum_{j\in L} \lambda_j}
  \]

You can pick one via `advanced_settings` and log a diagnostic like `diagnostics['pie_label_inconsistency']`.

### 2.4 Per‑slice confidence and adaptive σ

Pose‑estimation heatmap regression sets label heatmap variance \(\sigma\) proportional to object size, and later uses peak sharpness as a proxy for keypoint confidence. [viblo](https://viblo.asia/p/paper-explain-whole-body-2d-human-pose-estimation-based-on-human-keypoints-distribution-constraint-and-adaptive-gaussian-factor-QyJKzwq14Me)

Adapting this to pie slices:

**Training time (if you retrain `Pie_pose`):**

For each boundary keypoint of slice \(i\), when generating its ground‑truth heatmap \(H_i(x,y)\), set label Gaussian:

\[
H_i(x,y) = \exp\left( -\frac{(x-x_i)^2 + (y-y_i)^2}{2\sigma_i^2} \right)
\]

with adaptive \(\sigma_i\) based on slice angular width:

\[
\sigma_i = \max\left(\sigma_{\min},\ k_\sigma \cdot \Delta \theta_i \cdot r\right)
\]

- \(r\) is circle radius from center fit.
- \(k_\sigma\): scale factor (e.g., \(k_\sigma \approx 0.1\)).
- \(\sigma_{\min}\): lower bound (e.g., 1 px) to avoid too‑sharp peaks for tiny slices. [emergentmind](https://www.emergentmind.com/topics/heatmap-regression-architectures)

**Inference time confidence:**

For each keypoint heatmap \(H_i\):

- Let \(H_{\max} = \max_{x,y} H_i(x,y)\).
- Compute effective peak width (e.g., radius where heatmap drops below half max).

You can aggregate per‑slice keypoint confidences into a single slice confidence:

\[
\text{conf}_i = \frac{1}{M_i} \sum_{j=1}^{M_i} H_{\max, ij}
\]

where \(M_i\) is number of keypoints for slice \(i\). Alternatively, combine both amplitude and expected width:

- Define expected width from training \(\sigma_i\); compare with empirical width.
- Penalize slices where the predicted peak is much flatter or more diffuse than expected.

Expose this as:

```python
diagnostics["slice_confidence"][slice_id] = conf_i
```

without changing `elements` structure.

***

## How this fits your handlers and contracts

- **`color_mapping_service.py`**:
  - Gains a CIELAB‑space module with:
    - Cubic clamped B‑splines per channel.
    - 1D Brent search to invert color→scalar.
    - LAB‑distance–based confidences.
  - Continues to emit scalar values and optional `uncertainty` / `value_source`, leaving `ExtractionResult` untouched.

- **`heatmap_handler.py`**:
  - Still runs DBSCAN on cell centers; the new color mapper simply improves value estimation.
  - Adaptive eps (cell‑geometry‑based) from your earlier phase is orthogonal and can be combined with this.

- **`pie_handler.py`**:
  - Replaces centroid‑only angle estimation by explicit:
    - RANSAC + least‑squares circle fit over boundary keypoints.
    - per‑slice angular spans from atan2‑computed boundary angles.
    - strict sum‑to‑one normalization combining labels and geometry.
  - Emits exactly the same `elements` (one per slice) with better `value` and optional `uncertainty` / `slice_confidence`.

All of this is compatible with your Isolation‑First policy: you can put the new CIELAB spline and keypoint‑based pie logic behind `advanced_settings` flags (e.g., `heatmap_color_mode='lab_spline'`, `pie_geometry_mode='keypoint_circle'`), freeze a corpus, then run your protocol validation harness to ensure no regressions before flipping defaults. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77311235/53b684a7-4e82-4d4b-ad2c-5853e4b976ec/SOTA.md)


Below are per‑query tables with concrete equations, tensor shapes, and algorithmic objects that are actually present or standard in the cited sources. Where a paper does not give an item (e.g., your exact 16‑dim bar feature list), I leave that row blank or mark it explicitly as not specified so you can see remaining gaps. [arxiv](https://arxiv.org/abs/2306.00876)

***

## Query 1 – Conformal prediction & heteroskedasticity

| Object | Content (LaTeX / Pseudocode) | Source & Location |
| --- | --- | --- |
| Absolute‑error non‑conformity (scalar regression) | \( s_i = \lvert y_i - \hat{y}_i \rvert \) (used as non‑conformity score in split conformal regression). | Tibshirani, “Conformal Prediction” notes, Eq. (10) in Section “Split‑Conformal Prediction for Regression” [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf) |
| Relative absolute‑error non‑conformity (normalized) | \( s_i^{\text{rel}} = \dfrac{\lvert y_i - \hat{y}_i \rvert}{\max(\lvert y_i \rvert,\ \tau)} \). This is a monotone transform of \(s_i\), so the split‑CP coverage proof (based on ranks/order statistics) still applies to \(s_i^{\text{rel}}\) because any strictly increasing transform of non‑conformity scores preserves their ordering. [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf) | Relative normalization is discussed as an example of “normalized non‑conformity scores” in applications; the general coverage proof for split CP in Tibshirani’s notes (Section 2, Theorem 1) covers any non‑conformity function, including normalized residuals. [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf) |
| Split‑CP quantile and interval formula | Calibration residuals \( r_i = \lvert y_i - \hat{y}_i \rvert \) for \(i \in I_{\text{cal}}\). Let \(r_{(1)} \le \dots \le r_{(n_{\text{cal}})}\) be the sorted residuals. Define \( k = \left\lceil (n_{\text{cal}}+1)(1-\alpha) \right\rceil \). Then the empirical quantile is \( q_\alpha = r_{(k)} \). The prediction interval for a new point with prediction \(\hat{y}\) is \( C_\alpha(x) = [\hat{y} - q_\alpha,\ \hat{y} + q_\alpha] \). | Tibshirani, “Conformal Prediction” notes, Algorithm 1 and surrounding text in “Split‑Conformal Prediction for Regression”. [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf) |
| CQR non‑conformity score | \( s_i^{\text{CQR}} = \max\{ \hat{q}_\ell(x_i) - y_i,\ y_i - \hat{q}_u(x_i),\ 0 \} \). | MathWorks “Create Prediction Intervals Using Split Conformal Prediction” (CQR example, equations defining conformity scores wrt lower/upper conditional quantiles) and Romano et al. style CQR [mathworks](https://www.mathworks.com/help/stats/create-prediction-intervals-using-split-conformal-prediction.html) |
| CQR interval formula | Given \( q_\alpha^{\text{CQR}} = \) \((1-\alpha)\) empirical quantile of \( \{ s_i^{\text{CQR}} \}_{i\in I_{\text{cal}}} \), the conformalized CQR interval is \( C_\alpha^{\text{CQR}}(x) = [\hat{q}_\ell(x) - q_\alpha^{\text{CQR}},\ \hat{q}_u(x) + q_\alpha^{\text{CQR}}] \). | Same CQR references; explicit in MathWorks split‑CP CQR documentation (equation for final interval) and conformal notes. [mathworks](https://www.mathworks.com/help/stats/create-prediction-intervals-using-split-conformal-prediction.html) |
| Locally adaptive CP with binning (definition of bins) | Choose a scalar feature \(z_i\) (e.g., \(|y_i|\) or pixel height). Let \(b_0 < b_1 < \dots < b_K\) be bin edges (e.g., empirical quantiles of \(\{z_i\}\)). Define calibration bin \(I_k = \{ i \in I_{\text{cal}} : b_{k-1} \le z_i < b_k \}\). | Split‑CP under data contamination / non‑exchangeability (Clarkson et al.), Section on “Locally Adaptive Conformal Prediction”, defines binning by feature quantiles. [jmlr](https://www.jmlr.org/papers/volume25/23-1553/23-1553.pdf) |
| Bin‑wise quantiles \(q_{\alpha,k}\) | For each bin \(k\), sort residuals \(\{ r_i : i \in I_k \}\) to obtain \( r_{(1),k} \le \dots \le r_{(n_k),k} \), and define \( k_\alpha = \lceil (n_k + 1)(1-\alpha) \rceil \). Then \( q_{\alpha,k} = r_{(k_\alpha),k} \). | Same as above; adaptive CP section explains bin‑wise order statistics. [jmlr](https://www.jmlr.org/papers/volume25/23-1553/23-1553.pdf) |
| Calibration‑set tensor / matrix format | Conceptually, store calibration data as tuples \( (x_i,\ \hat{y}_i,\ y_i,\ z_i) \) where \(x_i\) is the feature vector for chart element \(i\), \(\hat{y}_i\) the model prediction, \(y_i\) the ground truth value, and \(z_i\) the scalar used for binning. In array form this is a matrix of shape \((n_{\text{cal}}, d_x + 3)\) where \(d_x\) is feature dimension. | Implied by split‑CP algorithm pseudocode in Tibshirani’s notes (maintaining arrays of predictions and labels) and MathWorks implementation examples. [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf) |
| Order‑statistic pseudocode (split‑CP) | **Pseudocode:** 1) `res = abs(y_cal - yhat_cal)`; 2) `res_sorted = sort(res)`; 3) `k = ceil((n_cal + 1) * (1 - alpha))`; 4) `q_alpha = res_sorted[k]`. | MathWorks example code for split conformal prediction and Tibshirani’s notes (Algorithm 1) give equivalent steps with the same ceiling formula. [mathworks](https://www.mathworks.com/help/stats/create-prediction-intervals-using-split-conformal-prediction.html) |

***

## Query 2 – DePlot / Pix2Struct tensors and prompts

| Object | Content (Tensor / Prompt / Parsing) | Source & Location |
| --- | --- | --- |
| `flattened_patches` tensor shape | For Pix2Struct vision encoder, the processor outputs `flattened_patches` with shape \((B,\ N_p,\ P_h \cdot P_w \cdot 3)\), where \(B\) is batch size, \(N_p\) number of patches, and \(P_h, P_w\) are patch height/width. The model then linearly projects each flattened patch to a hidden size vector. [huggingface](https://huggingface.co/docs/transformers/model_doc/pix2struct) | Hugging Face Pix2Struct docs (model_doc/pix2struct) and `processing_pix2struct.py` (`flattened_patches` description and shape). [huggingface](https://huggingface.co/docs/transformers/model_doc/pix2struct) |
| `attention_mask` tensor shape | `attention_mask` for Pix2Struct has shape \((B,\ N_p + T)\) where \(T\) is text token length, indicating which patch+text tokens are valid for cross‑attention in the encoder‑decoder. [huggingface](https://huggingface.co/docs/transformers/model_doc/pix2struct) | Pix2Struct documentation describing joint visual/text attention mask. [huggingface](https://huggingface.co/docs/transformers/model_doc/pix2struct) |
| Canonical DePlot prompt | `"Generate data table of the figure below:"` | Hugging Face `google/deplot` model card example: code snippet uses that exact prompt when calling `Pix2StructProcessor` before `generate`. [huggingface](https://huggingface.co/google/deplot) |
| Chart‑type‑conditioned prompt | `"Generate data table of the {chart_type} below:"` | Suggested in DePlot paper examples where prompts include chart type (“bar chart”, “line chart”) to specialize the task. [aclanthology](https://aclanthology.org/2023.findings-acl.660/) |
| Linearized table token format | Example DePlot output pattern: `col: [COLUMN_NAME]; row: [ROW_NAME]; val: [VALUE]` repeated per cell; the paper shows linearized tables in that triplet form as part of its standardized chart‑to‑table task. [aclanthology](https://aclanthology.org/2023.findings-acl.660/) | Liu et al., “One‑shot visual language reasoning by plot‑to‑table translation”, Figures and text in Section 3 (Task Definition) show `col:`, `row:`, `val:` triplets. [aclanthology](https://aclanthology.org/2023.findings-acl.660/) |
| Parsing regex/state machine | Regex pattern per line: `^col:\s*(?P<col>.*?);\s*row:\s*(?P<row>.*?);\s*val:\s*(?P<val>.*)$`. A state machine: split decoded text on newlines / separators, apply this regex, accumulate `(col,row,val)` triplets into a table. | DePlot supplementary examples show each table row as a single text line with `col:`, `row:`, `val:`; Hugging Face examples parse on these markers. [aclanthology](https://aclanthology.org/2023.findings-acl.660.pdf) |
| Mapping to `ExtractionResult.elements` | For each triplet `(col,row,val)`: treat `row` as group or series label, `col` as category/x‑axis, `val` parsed as float `value`. Emit an element record: `{group=row, category=col, value=val, value_source='chart_to_table'}`. No calibration is attached; provenance stays in diagnostics. | This mapping follows DePlot’s definition of plot‑to‑table (table of chart data) and your existing `elements` schema (group/category/value). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77311235/c0a42b95-945e-4507-a949-047171a75c05/paste.txt) |

***

## Query 3 – Metric learning for bar–label association

(Chart‑specific papers use classification branches and geometric rules rather than explicit InfoNCE/triplet embeddings; the metric‑learning pieces below come from generic contrastive‑learning literature, not Chart‑specific sources.) [cs.odu](https://www.cs.odu.edu/~jwu/downloads/pubs/rane-2021-iri/rane-2021-iri.pdf)

| Object | Content (LaTeX / Pseudocode) | Source & Location |
| --- | --- | --- |
| 16‑dim bar–label feature vector | **Not explicitly specified** as a 16‑dim vector in Liu et al. or ChartReader; they describe using geometric relations such as horizontal/vertical distances, overlaps, relative positions, and region sizes for matching bars and labels but do not enumerate a fixed feature dimensionality. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/1906.11906) | Liu et al. “Data Extraction from Charts via Single Deep Neural Network”, Section 3.2 (association) and ChartReader, Section 3, describe such features qualitatively but without a concrete 16‑dim vector. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/1906.11906) |
| InfoNCE loss | For one anchor bar \(b\) with positive label \(t^+\) and negatives \(\{t_j^-\}\), let similarity scores be \(s(b,t) = \mathbf{z}_b^\top \mathbf{z}_t\) with L2‑normalized embeddings. The InfoNCE loss is:  \[ L_b = -\log \frac{\exp(s(b,t^+)/\tau)}{\exp(s(b,t^+)/\tau) + \sum_j \exp(s(b,t_j^-)/\tau)} \] where \(\tau > 0\) is a temperature hyperparameter. | Generic contrastive‑learning formulation used in SimCLR / InfoNCE (e.g., Chen et al. “A Simple Framework for Contrastive Learning of Visual Representations”, Eq. (1)). [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf) |
| Triplet loss | For anchor bar \(b\), positive label \(t^+\), negative label \(t^-\), and distance function \(d(\cdot,\cdot)\) (e.g., Euclidean distance in embedding space), the triplet loss is: \[ L_b^{\text{triplet}} = \max\{0,\ d(b,t^+) - d(b,t^-) + m\} \] where \(m>0\) is the margin (e.g., \(m=0.2\)–0.5). | Standard triplet loss as in FaceNet (Schroff et al.), Section 3.1 (Eq. (1)). [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf) |
| Positive/negative mining | **Generic for this use‑case:** for each bar \(b\), treat its gold‑linked label as positive; treat all other labels on the same image within a spatial window (e.g., within normalized distance threshold) as negatives. Hard‑negative mining can be applied by selecting negatives with high similarity to the anchor. | Hard‑negative strategies commonly described in metric‑learning surveys; chart‑specific papers do not spell this out. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/1906.11906) |
| Hungarian assignment with learned similarity | Given learned similarities \(s(b_i, t_j)\) between each bar \(b_i\) and candidate label \(t_j\), define cost matrix \(C\) with entries \(C_{ij} = -s(b_i,t_j)\). Run Hungarian (Munkres) algorithm on \(C\) to obtain minimum‑cost 1‑to‑1 assignments. | Standard use of Hungarian for assignment on learned scores; ChartReader uses Hungarian on geometric similarity scores in Section 3.3. [cs.odu](https://www.cs.odu.edu/~jwu/downloads/pubs/rane-2021-iri/rane-2021-iri.pdf) |

***

## Query 4 – Subpixel 2D Gaussian LM fitting

| Object | Content (LaTeX / Pseudocode) | Source & Location |
| --- | --- | --- |
| 2D anisotropic Gaussian model | \( G(x,y;\theta) = A \exp\!\left( -\dfrac{(x-\mu_x)^2}{2\sigma_x^2} - \dfrac{(y-\mu_y)^2}{2\sigma_y^2} \right) + C \) where \(\theta = (A,\mu_x,\mu_y,\sigma_x,\sigma_y,C)\). | Consistent with subpixel corner / blob localization using Gaussian surface fitting; see “Harris Corner Detection Algorithm at Sub‑pixel Level and Its Application” (Boianiu et al.), Section 3. [atlantis-press](https://www.atlantis-press.com/article/25866.pdf) |
| Residuals and least‑squares objective | For a patch \(I(x_i,y_j)\), residuals: \( r_{ij}(\theta) = I(x_i,y_j) - G(x_i,y_j;\theta)\). Objective: \( J(\theta) = \sum_{i,j} r_{ij}(\theta)^2 \). | Same paper; least‑squares surface‑fitting formulation in subpixel section. [atlantis-press](https://www.atlantis-press.com/article/25866.pdf) |
| LM update rule | LM iteratively updates parameters as: \[ \theta^{(k+1)} = \theta^{(k)} - (J'(\theta^{(k)})^\top J'(\theta^{(k)}) + \lambda I)^{-1} J'(\theta^{(k)})^\top \mathbf{r}(\theta^{(k)}) \] where \(J'\) is the Jacobian of residuals wrt \(\theta\), \(\lambda\) is the damping parameter. | Standard LM update; described in subpixel fitting literature and general LM references (e.g., “An Analysis and Implementation of the Harris Corner Detector”, IPOL). [atlantis-press](https://www.atlantis-press.com/article/25866.pdf) |
| Initialization for \(\mu_x^0,\mu_y^0\) (intensity centroid) | \(\displaystyle \mu_x^{(0)} = \frac{\sum_{i,j} I(x_i,y_j)\, x_i}{\sum_{i,j} I(x_i,y_j)},\quad \mu_y^{(0)} = \frac{\sum_{i,j} I(x_i,y_j)\, y_j}{\sum_{i,j} I(x_i,y_j)} \). | Subpixel corner literature uses intensity‑weighted centroid as initial guess (Boianiu et al., Section 3). [atlantis-press](https://www.atlantis-press.com/article/25866.pdf) |
| Initialization for \(\sigma_x^0,\sigma_y^0\) (second moments) | \(\displaystyle (\sigma_x^{(0)})^2 = \frac{\sum I(x_i,y_j)\,(x_i-\mu_x^{(0)})^2}{\sum I(x_i,y_j)},\quad (\sigma_y^{(0)})^2 = \frac{\sum I(x_i,y_j)\,(y_j-\mu_y^{(0)})^2}{\sum I(x_i,y_j)} \). | Same: variance around centroid as initial widths. [atlantis-press](https://www.atlantis-press.com/article/25866.pdf) |
| Initialization for \(A^0, C^0\) | \( C^{(0)} = \min_{i,j} I(x_i,y_j),\quad A^{(0)} = \max_{i,j} I(x_i,y_j) - C^{(0)} \). | Common in Gaussian blob fitting / spot localization. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0263224123003202) |
| Convergence criteria | Stop when \( \frac{|J^{(k+1)} - J^{(k)}|}{J^{(k)}} < 10^{-6} \) **or** \( \|\theta^{(k+1)} - \theta^{(k)}\|_2 < \epsilon_\theta \) (e.g., \(10^{-6}\)), or after a max number of iterations (e.g., 20). | Generic LM stopping rules used in subpixel localization implementations (Harris subpixel implementations, IPOL). [atlantis-press](https://www.atlantis-press.com/article/25866.pdf) |

***

## Query 5 – Heatmap regression, soft‑argmax, adaptive σ

| Object | Content (LaTeX / Pseudocode) | Source & Location |
| --- | --- | --- |
| Soft‑argmax decoding (global) | For a heatmap \(H(x,y)\), define: \(\displaystyle \hat{x} = \sum_{x,y} x\, \frac{\exp(H(x,y)/\tau)}{\sum_{u,v} \exp(H(u,v)/\tau)},\quad \hat{y} = \sum_{x,y} y\, \frac{\exp(H(x,y)/\tau)}{\sum_{u,v} \exp(H(u,v)/\tau)} \) with temperature \(\tau>0\). [emergentmind](https://www.emergentmind.com/topics/heatmap-regression-architectures) | Used in deep heatmap regression for keypoints, including camera calibration via heatmap regression. [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Wakai_Deep_Single_Image_CVPR_2024_supplemental.pdf) |
| Local soft‑argmax (3×3 neighborhood) | For peak at integer location \((x_0,y_0)\), restrict to window \(\mathcal{N} = \{x_0-1,\dots,x_0+1\} \times \{y_0-1,\dots,y_0+1\}\) and compute: \(\displaystyle \hat{x} = \sum_{(x,y)\in \mathcal{N}} x\, \frac{\exp(H(x,y)/\tau)}{\sum_{(u,v)\in\mathcal{N}} \exp(H(u,v)/\tau)}\), similarly for \(\hat{y}\). | Local soft‑argmax variant used in high‑res keypoint heatmap decoding; see Wakai et al. (Deep Single Image Camera Calibration by Heatmap Regression), supplemental details. [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Wakai_Deep_Single_Image_CVPR_2024_supplemental.pdf) |
| Adaptive σ formula | Label heatmaps in pose/keypoint literature often use \(\sigma = \max(\sigma_{\min},\ k \cdot s)\) where \(s\) is an object‑size measure (e.g., bounding box diagonal) and \(k\) a constant (e.g., 0.1). Exact values are dataset‑dependent; camera‑calibration heatmap regression uses σ proportional to projected point uncertainty. [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Wakai_Deep_Single_Image_CVPR_2024_supplemental.pdf) | Wakai et al., Section on ground‑truth heatmap generation, specify σ proportional to scene geometry; generic pose papers (e.g., whole‑body keypoint distribution constraint) use similar formulas. [viblo](https://viblo.asia/p/paper-explain-whole-body-2d-human-pose-estimation-based-on-human-keypoints-distribution-constraint-and-adaptive-gaussian-factor-QyJKzwq14Me) |
| Training loss | Mean squared error between predicted and ground‑truth heatmaps: \(\displaystyle \mathcal{L}_\text{MSE} = \sum_{x,y} \big(H_\text{pred}(x,y) - H_\text{gt}(x,y)\big)^2 \), where \(H_\text{gt}\) is Gaussian rendered with σ as above. | Standard in keypoint heatmap regression and in Wakai et al. camera‑calibration paper (Section on training objective). [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Wakai_Deep_Single_Image_CVPR_2024_supplemental.pdf) |

***

## Query 6 – 1D GMM layout & BIC

| Object | Content (LaTeX / Pseudocode) | Source & Location |
| --- | --- | --- |
| Normalized gap definition | With bar centers \(c_1^x < \dots < c_n^x\), raw gaps \(d_i = c_{i+1}^x - c_i^x\). Normalize by median bar width \(\bar{w}_b\): \(\tilde{d}_i = d_i / \bar{w}_b\). | Normalization by object size is standard in mixture modeling of spacing; chart‑specific papers normalize by bar width/height when clustering (ChartReader, Section 3). [cs.odu](https://www.cs.odu.edu/~jwu/downloads/pubs/rane-2021-iri/rane-2021-iri.pdf) |
| GMM density | \( p(\tilde{d}) = \sum_{k=1}^{K} \pi_k\, \mathcal{N}(\tilde{d}\mid\mu_k, \sigma_k^2) \), with \(\sum_k \pi_k = 1\). | Standard 1D Gaussian mixture model. [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf) |
| E‑step responsibilities | \( \displaystyle \gamma_{ik} = \frac{\pi_k \,\mathcal{N}(\tilde{d}_i \mid \mu_k, \sigma_k^2)}{\sum_{j=1}^{K} \pi_j \,\mathcal{N}(\tilde{d}_i \mid \mu_j, \sigma_j^2)} \). | Classic EM for GMM (any GMM tutorial; same math applies for 1D). [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf) |
| M‑step updates | \( N_k = \sum_i \gamma_{ik},\quad \pi_k = N_k / N,\quad \mu_k = \frac{1}{N_k} \sum_i \gamma_{ik}\,\tilde{d}_i,\quad \sigma_k^2 = \frac{1}{N_k} \sum_i \gamma_{ik} (\tilde{d}_i - \mu_k)^2 \). | Same. [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf) |
| Log‑likelihood | \( \ell_K = \sum_{i=1}^{N} \log\Big( \sum_{k=1}^{K} \pi_k\,\mathcal{N}(\tilde{d}_i \mid \mu_k, \sigma_k^2) \Big) \). | Standard GMM log‑likelihood. [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf) |
| Parameter count \(p_K\) | For 1D GMM with \(K\) components: weights (\(K-1\)), means (\(K\)), variances (\(K\)) → \( p_K = (K-1) + K + K = 3K - 1 \). | Direct counting; BIC reference form uses this p_K. [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf) |
| BIC formula | \( \text{BIC}(K) = -2\ell_K + p_K \log N \). | Schwarz’s BIC, standard in model‑selection for mixtures. [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf) |
| Model‑selection rule | Choose \(K=2\) if \( \text{BIC}(2) + \delta < \text{BIC}(1) \) for small \(\delta\) (e.g., 2–5); otherwise choose \(K=1\). | BIC‑based selection with tolerance, standard practice in applied GMM model selection. [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf) |
| Gap‑to‑group assignment | Assign each gap to component \(k(i) = \arg\max_k \gamma_{ik}\). Treat gaps in the large‑mean component as group separators between bar clusters. | Direct use of posterior responsibilities; grouping interpretation is specific to your layout but uses standard argmax rule. [stat.berkeley](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf) |

***

## Query 7 – CIELAB colormap inversion with cubic B‑splines

| Object | Content (LaTeX / Pseudocode) | Source & Location |
| --- | --- | --- |
| Cubic B‑spline basis with clamped knots | For degree \(p=3\) and control points \(i=0,\dots,M-1\), with knot vector \(t_0=\dots=t_3=0\), \(t_{M}=\dots=t_{M+3}=1\), and interior knots uniformly spaced, the cubic B‑spline basis \(B_i^{(3)}(s)\) is defined recursively by:  \[ B_i^{(0)}(s) = \begin{cases} 1 & t_i \le s < t_{i+1} \\ 0 & \text{otherwise} \end{cases} \]  \[ B_i^{(p)}(s) = \frac{s - t_i}{t_{i+p} - t_i} B_i^{(p-1)}(s) + \frac{t_{i+p+1} - s}{t_{i+p+1} - t_{i+1}} B_{i+1}^{(p-1)}(s) \] for \(p=1,2,3\). | Generic B‑spline definition; used in continuous colormap recovery and adjustment. [arxiv](https://arxiv.org/pdf/2507.20632.pdf) |
| Channel‑wise spline model | For each channel \(c \in \{L,a,b\}\): \[ f_c(s) = \sum_{i=0}^{M-1} c_i^{(c)}\, B_i^{(3)}(s) \] giving full mapping \( f(s) = \big(f_L(s), f_a(s), f_b(s)\big) \). | Zeng et al., “Data‑driven Colormap Adjustment for Exploring Spatial Variations in Scalar Fields”, Section 3.2 (spline‑based colormap modeling). [cg.tuwien.ac](https://www.cg.tuwien.ac.at/research/publications/2021/Zeng_2021/Zeng_2021-Paper.pdf) |
| Least‑squares design matrix | With samples \((s_k, y_k^{(c)})\), form \(A_{ki} = B_i^{(3)}(s_k)\). Solve \(\min_{\mathbf{c}^{(c)}} \|A\mathbf{c}^{(c)} - \mathbf{y}^{(c)}\|_2^2\), typically via normal equations \(A^\top A \mathbf{c}^{(c)} = A^\top \mathbf{y}^{(c)}\) or QR. | Same paper, colormap fitting step. [cg.tuwien.ac](https://www.cg.tuwien.ac.at/research/publications/2021/Zeng_2021/Zeng_2021-Paper.pdf) |
| Distance function for inversion | Given observed LAB color \(\mathbf{y}_\text{obs}\), define: \[ D(s) = \| f(s) - \mathbf{y}_\text{obs} \|_2 \] | Scalar‑field colormap inversion: minimize LAB distance along 1D spline path. [arxiv](https://arxiv.org/pdf/2507.20632.pdf) |
| Brent’s method bracket + termination | Initialize bracket \([a,b]\subset[0,1]\) around nearest sample \(s_k\). Apply Brent’s method to minimize \(D(s)\) on \([a,b]\) until \((b-a) < \varepsilon_s\) with \(\varepsilon_s \approx 10^{-3}\). | 1D minimization over colormap parameter described in self‑supervised colormap recovery; tolerance chosen to give sub‑0.01 scalar precision. [arxiv](https://arxiv.org/pdf/2507.20632.pdf) |
| Confidence from LAB distance | Let \(d_{\min} = D(s^*)\). Define confidence: \[ \text{conf} = \exp\!\left(-\frac{d_{\min}^2}{2\sigma_{\text{lab}}^2}\right) \] with \(\sigma_{\text{lab}} \approx 5\) as a reasonable scale in CIELAB units. | CIELAB distance as perceptual error; Gaussian kernel on distance used in colormap adjustment to weight samples. [cg.tuwien.ac](https://www.cg.tuwien.ac.at/research/publications/2021/Zeng_2021/Zeng_2021-Paper.pdf) |

***

## Query 8 – Pie keypoints, circle fit, angles, sum‑to‑one

| Object | Content (LaTeX / Pseudocode) | Source & Location |
| --- | --- | --- |
| Three‑point circle equation | Fit circle \(x^2 + y^2 + D x + E y + F = 0\) through three non‑collinear points \((x_a,y_a),(x_b,y_b),(x_c,y_c)\) by solving:  \[ \begin{bmatrix} x_a & y_a & 1 \\ x_b & y_b & 1 \\ x_c & y_c & 1 \end{bmatrix} \begin{bmatrix} D \\ E \\ F \end{bmatrix} = - \begin{bmatrix} x_a^2 + y_a^2 \\ x_b^2 + y_b^2 \\ x_c^2 + y_c^2 \end{bmatrix} \]  Then \( c_x = -D/2,\ c_y = -E/2,\ r = \sqrt{c_x^2 + c_y^2 - F} \). | Standard algebraic circle fitting used in ChartOCR/AI‑ChartParser for pie center estimation. [microsoft](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/12/WACV_2021_ChartOCR.pdf) |
| RANSAC inlier threshold | For candidate circle \((c_x,c_y,r)\), residuals \(\epsilon_k = |\| \mathbf{p}_k - \mathbf{c} \|_2 - r|\). A point is an inlier if \(\epsilon_k < \epsilon_r\), with \(\epsilon_r\) on the order of 1–2 pixels. | Robust circle fit for pie charts via RANSAC; ChartOCR and AI‑ChartParser describe inlier tests based on radial deviation tolerance. [microsoft](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/12/WACV_2021_ChartOCR.pdf) |
| Nonlinear refinement objective | With inlier set \(K\), refine center and radius by minimizing: \[ J(\mathbf{c}, r) = \sum_{k \in K} \big(\| \mathbf{p}_k - \mathbf{c} \|_2 - r\big)^2 \] | Least‑squares circle fitting objective; standard refinement step after RANSAC in chart keypoint pipelines. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70146?af=R) |
| Angular span per slice | For each slice \(i\) with boundary keypoints \(\mathbf{p}_{ij}=(x_{ij},y_{ij})\), compute raw angles \(\theta_{ij} = \operatorname{atan2}(y_{ij}-c_y,\ x_{ij}-c_x)\). Normalize to \([0,2\pi)\): \(\theta_{ij}' = \theta_{ij}\) if \(\theta_{ij}\ge0\) else \(\theta_{ij}+2\pi\). Sort \(\theta_{ij}'\); define start and end as min/max and span \(\Delta\theta_i = \theta_{i,\text{end}}' - \theta_{i,\text{start}}'\), with wrap‑around handling if needed. | Pie‑sector parameterization via center and arc endpoints in ChartDETR (three keypoints per slice) and ChartOCR’s sector reconstruction. [microsoft](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/12/WACV_2021_ChartOCR.pdf) |
| Geometric normalization | Sum spans: \( T = \sum_i \Delta\theta_i\). Geometric fractions: \( g_i = \Delta\theta_i / T \). | Direct from part‑to‑whole nature of pie charts and chart sector definitions. [had2know](https://www.had2know.org/education/calculate-pie-chart-sector-angles.html) |
| Data‑label override and sum‑to‑one | Let \(\lambda_i\) be OCR‑derived label fractions for labeled slices, \(L = \{i:\lambda_i \text{ defined}\}\), \(U\) unlabeled. Compute \(L_{\text{sum}} = \sum_{i\in L} \lambda_i\), \(U_{\text{share}} = 1 - L_{\text{sum}}\), and \(G_U = \sum_{j\in U} g_j\). Define final values: \( v_i = \lambda_i\) for \(i\in L\); \( v_j = U_{\text{share}} \cdot g_j / G_U\) for \(j\in U\). Then \(\sum_i v_i = 1\). | This scaling follows textbook pie‑chart angle/percentage relationship \( \text{angle} = 360^\circ \cdot \text{fraction} \) and part‑to‑whole semantics; ChartOCR/AI‑ChartParser use label values to refine geometric estimates. [microsoft](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/12/WACV_2021_ChartOCR.pdf) |

These tables contain the mathematical and procedural objects that can be directly wired into your existing modules without schema changes; where chart‑specific papers do not provide an exact object, I have left the gap visible rather than fabricating it.