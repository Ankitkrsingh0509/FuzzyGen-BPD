# Credit Card Fraud Detection ML Project

This project trains a machine learning model to detect suspicious credit card transactions. It includes:

- A synthetic transaction dataset generator with class imbalance similar to real fraud problems
- A training pipeline built with scikit-learn
- A Kaggle-ready notebook for experimenting in the Kaggle environment
- Evaluation metrics such as ROC AUC, average precision, confusion matrix, and a classification report

## Project structure

```text
.
├── README.md
├── notebooks
│   └── credit_card_fraud_detection_kaggle.ipynb
├── requirements.txt
├── scripts
│   ├── generate_sample_data.py
│   └── train_model.py
└── src
    └── fraud_detection
        ├── __init__.py
        ├── data.py
        ├── evaluate.py
        └── model.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the project

1. Generate a sample dataset:

```bash
python3 scripts/generate_sample_data.py
```

2. Train the fraud detection model:

```bash
python3 scripts/train_model.py
```

## Outputs

- `data/credit_card_transactions.csv`: generated dataset
- `models/fraud_detector.joblib`: trained model pipeline
- `reports/metrics.json`: saved evaluation report

## Run on Kaggle

Open [notebooks/credit_card_fraud_detection_kaggle.ipynb](/Users/ankitkumarsingh/Documents/New%20project/notebooks/credit_card_fraud_detection_kaggle.ipynb) and upload it to Kaggle as a new notebook.

The notebook supports two dataset formats:

- Kaggle's common fraud dataset with target column `Class`
- This project's local sample dataset with target column `is_fraud`

If you attach a Kaggle dataset such as `creditcard.csv`, the notebook will use that automatically from `/kaggle/input/...`. Otherwise it falls back to the sample CSV in this repo.

The Kaggle notebook now also includes:

- ROC and precision-recall curve plots
- A confusion matrix heatmap
- Automatic XGBoost training when Kaggle has `xgboost` available
- Threshold comparison for fraud-alert tuning
- Feature importance charts
- `submission.csv` export when a competition-style `test.csv` file exists

## FuzzyGen-BPD Streamlit prototype

The separate `fuzzygen_bpd_app.py` app implements the paper's fuzzy Behavioral Patch Distance scoring and its build, PoV, and functional-test eligibility gates. It includes the paper's worked example and accepts a CSV of candidate features and gate outcomes.

```bash
pip install -r requirements.txt
streamlit run fuzzygen_bpd_app.py
```

CSV columns: `vulnerability_id`, `candidate_id`, `TestDelta`, `CoverageDelta`, `DiffSize`, `build_pass`, `pov_pass`, and `functional_pass`. Feature values must be in `[0, 1]`; gate fields accept true/false or 1/0. The app ranks eligible candidates by ascending BPD and exports scored rows. It does not build or run patches, generate tests, measure JaCoCo coverage, or establish patch correctness. The source paper reports synthetic demonstration results only and leaves Vul4J experiments incomplete; the app presents no real-benchmark performance claims.

### Run the paper's synthetic evaluation pipeline

```bash
python scripts/run_fuzzygen_bpd_pipeline.py
```

The run creates held-out predictions, fold scores, learned weights, metric JSON, and interactive Plotly figures under `reports/fuzzygen_bpd/`. It uses the synthetic feature/label distributions described in the paper with five program-grouped outer folds, a four-fold inner search for fuzzy weights and threshold, a 60-evaluation random search, and an equal-weight baseline. It reports a program-cluster bootstrap interval and paired program-level sign-flip comparison. This validates an implementation path only; it is not an evaluation on PatchEval or Vul4J, and it does not compare all five optimizer families in the infographic.

### Compare all five optimizers

```bash
python scripts/compare_fuzzygen_optimizers.py
```

This runs Random Search, GA, TPE, GP-BO, and SMAC-style random-forest expected improvement at an equal 60 objective evaluations, using the same grouped folds, feature data, and threshold/weight encoding. It also includes an equal-weight BPD baseline with its decision threshold tuned only on inner folds. Outputs include per-candidate predictions, learned weights, held-out confusion matrices, search traces, grouped-fold metrics, and paired program-level Wilcoxon tests with Holm correction in the same reports directory. Lightweight TPE, GP-BO, and SMAC-style implementations are described in the generated `optimizer_comparison.json`; they are not claimed to be bit-for-bit equivalents to the named external optimizer packages. Results remain synthetic demonstration evidence only.

### Generate the research figure collection

```bash
python scripts/generate_fuzzygen_research_figures.py
```

This creates the complete figure collection under `reports/fuzzygen_bpd/`: fuzzy membership profiles, inference transformation, weight-impact ranking, optimizer convergence, learned weights, the mean-versus-consistency trade-off, ablation, feature distributions, BPD-versus-label analysis, candidate ranking, held-out fold variation, membership sensitivity, and Macro-F1 comparison. The Streamlit app loads them under **Research figure collection**. It only generates figures supported by the synthetic evaluation artifacts currently available; real PatchEval or Vul4J datasets will produce separate results only after candidate patches, gate outcomes, measured feature values, and independent labels are available.

### Generate the paper figure collection

```bash
python scripts/generate_fuzzygen_research_figures.py
```

This creates Fig. 2–8 and six supporting interactive Plotly figures under `reports/fuzzygen_bpd/`; Fig. 1 uses the supplied research-framework infographic. The collection covers membership profiles, inference transformation, ranking sensitivity to weights, optimizer convergence, learned weights, the mean-versus-consistency trade-off, feature ablation, feature distributions, score/label relationship, ranking changes, held-out variability, membership-boundary sensitivity, and Macro-F1 comparison. Every analytical figure is marked as synthetic demonstration evidence.

## Use your own dataset

If you already have a dataset, replace `data/credit_card_transactions.csv` with your CSV and make sure it contains:

- Numeric feature columns
- A target column named `is_fraud` where `1` means fraud and `0` means non-fraud

Then run:

```bash
python3 scripts/train_model.py --data path/to/your_dataset.csv
```

## Future improvements

- Try XGBoost or LightGBM for stronger tabular performance
- Add threshold tuning for recall-heavy fraud operations
- Track experiments with MLflow
- Add a small Streamlit dashboard for live predictions
