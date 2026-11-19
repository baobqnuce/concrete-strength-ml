# Predicting Concrete Compressive Strength from Mix Design with Regularised Linear Regression

Machine Learning mini-project (CV7001). The goal is to predict the compressive strength
$f'_c$ (MPa) of a concrete specimen from its mix proportions and age. The models are
least squares, polynomial feature expansion, ridge and lasso, which are the linear-regression
methods taught in the course. A gradient-boosting model is included as a non-linear reference.

## Problem, data, inputs and outputs

| | |
|---|---|
| **Task** | Regression: estimate $f'_c$ before (or instead of) a destructive cylinder/cube test |
| **Dataset** | UCI *Concrete Compressive Strength* (Yeh, 1998), 1030 laboratory tests, 427 distinct mix designs |
| **Inputs (8)** | cement, blast-furnace slag, fly ash, water, superplasticiser, coarse aggregate, fine aggregate (kg/m³), age (days, 1–365) |
| **Output** | compressive strength (MPa, 2.3–82.6) |
| **Selected model** | `log(age)` → standardise → polynomial features (degree 2, 44 terms) → standardise → ridge regression |

The same mix design is tested at several ages (for example 3, 7, 28 and 90 days). Rows are
therefore **not independent**. All splits and cross-validation folds are **grouped by mix
design**, so no mix appears in both training and evaluation data. Section *Results*
shows how much a random split would overestimate performance.

## Installation

Tested with Python 3.13 on Linux.

```bash
git clone https://github.com/baobqnuce/concrete-strength-ml.git
cd concrete-strength-ml
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The data are downloaded automatically on first use into `data/concrete.csv`. This is a CSV mirror
of the UCI file, verified by SHA-256 checksum inside `src/data.py`.

- Original source: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength (`Concrete_Data.xls`)
- Mirror used: https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv
  (same 1030 rows; columns renamed to `cement, slag, ash, water, superplastic, coarseagg, fineagg, age, strength`)

No labels were created by us. The data are public (CC BY 4.0 on UCI).

## Train / reproduce all results

```bash
python -m src.train            # ~15 s: model selection, test evaluation, figures, saves the model
python -m src.repeat_splits    # ~1 min: repeats everything over 10 different grouped splits
```

Outputs:

| File | Content |
|---|---|
| `results/metrics.json` | selected hyper-parameters, test metrics of all models, leakage check |
| `results/model_comparison.csv` | validation and test metrics table |
| `results/cv_grid.csv` | grouped-CV RMSE for every (degree, alpha) pair |
| `results/repeated_splits.json` / `.csv` | mean ± std over 10 grouped splits |
| `results/figures/results.png` | (a) train/validation RMSE vs. polynomial degree; (b) predicted vs. measured (test) |
| `results/figures/cv_heatmap.png` | validation RMSE over degree × alpha |
| `models/best_model.joblib` | trained pipeline (scikit-learn 1.9.1) |

## Run inference on new inputs

```bash
python -m src.predict examples/new_mixes.csv                 # hypothetical mixes
python -m src.predict examples/test_samples.csv              # 5 held-out test rows with measured strength
python -m src.predict my_mixes.csv --out predictions.csv     # your own file
```

The input CSV needs the 8 input columns above. A `strength` column is optional; if it is
present, the script prints the error. Rows outside the training range are flagged as
extrapolation.

Example output (`examples/new_mixes.csv`, same mix at 7/28/90 days):

```
cement slag ash water superplastic coarseagg fineagg age predicted_strength_MPa
   350    0   0   175            5      1000     800   7                  26.16
   350    0   0   175            5      1000     800  28                  36.19
   350    0   0   175            5      1000     800  90                  43.32
```

## Evaluation procedure

1. **Grouped split:** 80 % development set (815 rows / 341 mixes) and 20 % test set (215 rows / 86 mixes),
   using `GroupShuffleSplit` with seed 42. The test set is used only once, at the end.
2. **Model selection:** grouped 5-fold CV (`GroupKFold`) on the development set over polynomial degree
   1–5 × 15 ridge penalties $\alpha \in [10^{-4}, 10^{3}]$. The model is selected by mean validation RMSE.
3. The selected configuration is refit on the full development set and evaluated on the test set
   (RMSE, MAE, $R^2$).
4. **Robustness check:** steps 1–3 are repeated for seeds 0–9.

Note: scikit-learn's `Ridge` minimises $\lVert \mathbf y - X\mathbf w\rVert^2 + \alpha\lVert\mathbf w\rVert^2$
(sum, not mean), so $\alpha = 2N\lambda$ relative to the course notation $\frac{1}{2N}\lVert\cdot\rVert^2 + \lambda\lVert\mathbf w\rVert^2$.

## Results

Test set of the seed-42 split:

| Model | Val. RMSE (grouped CV) | Test RMSE | Test MAE | Test $R^2$ |
|---|---|---|---|---|
| Mean predictor | 16.30 | 18.04 | 14.67 | 0.00 |
| OLS, raw age | 10.47 | 10.74 | 8.38 | 0.645 |
| OLS, log(age) | 7.30 | 7.36 | 5.42 | 0.833 |
| **Ridge, poly degree 2 (selected)** | **6.45** | 7.44 | 5.24 | 0.830 |
| Gradient boosting (reference) | 6.23 | 6.34 | 4.31 | 0.876 |

Over 10 grouped splits (mean ± std of test metrics):

| Model | RMSE (MPa) | MAE (MPa) | $R^2$ |
|---|---|---|---|
| OLS, log(age) | 7.34 ± 0.43 | 5.69 ± 0.39 | 0.80 ± 0.03 |
| Ridge, CV-selected (degree 2 in 9/10 splits) | 7.19 ± 0.91 | 5.19 ± 0.43 | 0.80 ± 0.06 |
| Gradient boosting | 5.93 ± 0.52 | 4.23 ± 0.51 | 0.87 ± 0.02 |

Leakage check (validation RMSE on the development set): random 5-fold CV gives 5.76 MPa for ridge
and 4.23 MPa for gradient boosting. Grouped 5-fold CV gives 6.45 and 6.23 MPa. A random split
therefore makes the error look up to about 32 % smaller than it is.

## Repository layout

```
src/data.py            download, checksum, mix-design groups, grouped split
src/models.py          pipelines (OLS / ridge / lasso with polynomial features; gradient boosting)
src/train.py           model selection, test evaluation, figures, saves model
src/repeat_splits.py   repeated grouped splits (robustness)
src/predict.py         inference on a CSV of new mixes
examples/              input files for inference
results/, models/      generated outputs (committed for convenience)
report/                2-page LaTeX report
```

## References and disclosures

- I.-C. Yeh, "Modeling of strength of high-performance concrete using artificial neural networks",
  *Cement and Concrete Research* 28(12):1797–1808, 1998. Dataset: UCI ML Repository, doi:10.24432/C5PK67.
- scikit-learn (Pedregosa et al., JMLR 12, 2011) was used for all models. No pretrained models were used.
- Generative AI (Claude, Anthropic) assisted in writing the code.
