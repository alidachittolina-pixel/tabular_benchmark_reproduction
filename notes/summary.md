# Paper Summary

**Title:** Why do tree-based models still outperform deep learning on typical tabular data?
**Authors:** Léo Grinsztajn, Edouard Oyallon, Gaël Varoquaux
**Venue:** NeurIPS 2022 (Datasets & Benchmarks Track)
**Link:** https://arxiv.org/abs/2207.08815
**Code (original):** https://github.com/LeoGrin/tabular-benchmark

---

## The four questions

### 1. What problem is the paper solving?
There was no standard, fair benchmark for tabular data, so competing claims about
whether deep learning matches or beats tree-based models could not be trusted:
evaluation methodology varied, datasets were few, and hyperparameter-tuning effort
was unequal between methods. The paper builds a rigorous benchmark to test whether
tree-based models still win on typical (medium-sized) tabular data, and then
investigates *why* they win.

**Note:** This is a benchmark / empirical study, NOT a new-model paper.

### 2. What dataset(s) does it use?
45 tabular datasets from varied domains, shared via OpenML, selected by explicit
criteria (heterogeneous columns, not high-dimensional, real-world, not too easy,
i.i.d., at least ~3000 samples, etc.). They are split into groupings:
- numerical-only vs. numerical + categorical features
- medium regime (training set truncated to 10,000 samples); large regime (50,000)
  is in the appendix

Preprocessing choices: missing values dropped; multi-class targets binarized
(two most frequent classes, balanced); low-cardinality categoricals only.

### 3. What model does it propose?
None. The contributions are (a) the benchmark + methodology and (b) three empirical
findings explaining why trees win:
1. Neural nets are biased toward overly smooth functions.
2. MLP-like nets are hurt by uninformative features.
3. Rotation invariance is undesirable for tabular data.

Models **compared** (not proposed):
- Tree-based: scikit-learn RandomForest, GradientBoosting / HistGradientBoosting,
  XGBoost
- Deep: MLP, ResNet, FT-Transformer, SAINT

### 4. How is performance evaluated?
- Metric: accuracy (classification), R² (regression).
- Normalization: per-dataset affine rescaling between the best model and a low
  quantile (10% for classification, 50% for regression) so scores can be averaged
  across datasets of differing difficulty.
- Fair tuning: performance reported as a function of the number of random-search
  iterations (~400 per model/dataset), with the search order reshuffled 15+ times
  to produce error bars. This accounts for hyperparameter-tuning cost.

---

## What "reproducing this paper" actually means for me
Rebuild the pipeline and re-run the tree-vs-deep comparison on a small subset of
their datasets:
1. Pull a few datasets from OpenML (start with numerical-only classification).
2. Apply their preprocessing (drop missing, binarize target, encode categoricals,
   truncate to 10K, Gaussianize features for NNs).
3. Random-search hyperparameters for each model.
4. Score with accuracy / R², normalize, aggregate across datasets.
5. Plot score vs. number of search iterations, with shuffled-order error bars.
6. Compare my ranking to the paper's (trees should stay on top).

## Implementation checklist (easiest -> hardest) — to refine in Step 4
- [ ] Set up environment (scikit-learn, xgboost, openml, pandas, numpy, matplotlib)
- [ ] Load 3-5 datasets from OpenML
- [ ] Preprocessing pipeline (missing data, binarize, encode, truncate, scale)
- [ ] Baseline: default-hyperparameter RandomForest / XGBoost vs. MLP
- [ ] Train/valid/test split matching the paper's protocol
- [ ] Random-search loop with a fixed iteration budget
- [ ] Metrics + per-dataset normalization
- [ ] Aggregation across datasets + score-vs-iterations plot
- [ ] Compare with the paper; write up differences
- [ ] (Extension) add LightGBM/CatBoost, a linear baseline, or an ablation

## Open questions / things to confirm while reading in depth
- Exact random-search spaces (paper Appendix A.3)
- Exact list of datasets used per setting (Appendix A.1)
- How they handle the validation split for model selection
