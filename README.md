# Recommender Systems Project - Group 3

---

## 1. Repository Structure

```text
recommender_systems_project/
├── README.md                              # Reproducibility instructions & architecture
├── requirements.txt                       # Explicit pinned dependencies
├── run.py                                 # Single documented command entry point
├── Report_Group3.docx                     # Project report
├── Results_Group3.xlsx                    # Official course results workbook
├── ml-1m/                                 # Local MovieLens dataset (ignored from Git)
└── part1/
    ├── __init__.py
    ├── data_loader.py                     # Data ingestion, encoding & schema validation
    ├── data_analysis.py                   # Descriptive statistics & characterization plots
    ├── filter_split.py                    # Threshold filtering & chronological per-user split
    ├── baselines.py                       # Most Popular (3 variants) & Item-based kNN models
    ├── metrics.py                         # Top-10 ranking metrics & full catalogue coverage
    ├── tune.py                            # Hyperparameter search on validation set
    └── outputs/
        ├── item_long_tail.png             # Output figures for report
        ├── user_activity_distribution.png
        ├── rating_distribution.png
        ├── temporal_activity.png
        ├── tuning_results.csv             # Validation grid search results
        └── results_part1.csv              # Standardized test results CSV
└── part2/
    ├── __init__.py
    ├── tune.py                            # Validation tuning (8 configs per model; standalone)
    ├── evaluation.py                      # Multi-seed test evaluation, 95% CIs & paired t-tests
    ├── disaggregation.py                  # User activity & item popularity breakdowns
    ├── sparsity.py                        # 100% / 50% / 20% training sparsity ablation
    ├── plots.py                           # Wide report-ready figures from saved CSVs
    ├── models/
    │   ├── __init__.py
    │   ├── base.py                        # Shared BaseRecommender interface
    │   ├── content_based.py               # Self-implemented genre + release-year recommender
    │   ├── mf_sgd.py                      # Surprise SVD (MF-SGD)
    │   ├── bpr.py                         # Implicit BPR-MF
    │   └── neural_vae.py                  # PyTorch Mult-VAE
    └── outputs/
        ├── tuning_results_part2.csv       # 32-config validation tuning grid
        ├── test_results_part2.csv         # Multi-seed test results (Mean ± 95% CI)
        ├── paired_tests_part2.csv         # Paired t-tests on per-user NDCG@10
        ├── disaggregation_part2.csv       # User activity & item popularity results
        ├── sparsity_part2.csv             # Sparsity ablation results (100%, 50%, 20%)
        ├── part2_accuracy_vs_coverage.png # Test NDCG@10 & Accuracy-Coverage trade-off
        └── part2_disagg_and_sparsity.png  # Item popularity breakdown & sparsity curve
```

---

## 2. Environment & Requirements

- **Programming Language**: Python 3.12+ (tested on Python 3.12 / 3.14 via WSL Ubuntu 24.04).
- **Random Seed**: `42` Is the primary seed. Stochastic Part 2 models are evaluated over seeds `[42, 43, 44, 45, 46]` for 95% confidence intervals.
- **Core Libraries**: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `tqdm`, `openpyxl`, `torch`, `implicit`, `scikit-surprise`.

### Installation
Clone the repository and install all dependencies:
```bash
git clone https://github.com/PauDuch/recommender_systems_project.git
cd recommender_systems_project
pip install -r requirements.txt
```

---

## 3. Dataset Setup

The project uses the standard **MovieLens 1M** dataset provided by GroupLens:
- The dataset folder should be placed in the project directory as `ml-1m` (e.g., `./ml-1m`).

The path to the dataset is fully configurable via the `--data-dir` argument.

---

## 4. How to Run the Project
To run the complete project and reproduce all data (plots, statistics, baseline evaluation, and CSV results), execute:

```bash
python run.py --data-dir ml-1m
```
*(Replace `ml-1m` with the relative or absolute path to your MovieLens 1M directory).*

### Running a Specific Part
A single part of the project can be executed using the `--part` argument:

```bash
# Run Part 1 (data analysis, filtering/split, baselines & test evaluation)
python run.py --data-dir ml-1m --part 1

# Run Part 2 (multi-seed test evaluation, disaggregation, sparsity ablation & plots)
python run.py --data-dir ml-1m --part 2
```

### What Is Not Included in `run.py` (Standalone Scripts)
Hyperparameter tuning is kept separate from `run.py` in both parts so that `run.py` reproduces the final test-set evaluations using the already-tuned configurations without re-running grid searches, which can take to much time:

- **Part 1:** Only `part1/tune.py` is standalone (`python part1/tune.py --data-dir ml-1m`).
- **Part 2:** `part2/tune.py` is standalone (`python part2/tune.py --data-dir ml-1m`). 

Unlike Part 1, each Part 2 stage can also be run individually if needed:
```bash
python part2/evaluation.py --data-dir ml-1m      # Multi-seed test evaluation + paired t-tests
python part2/disaggregation.py --data-dir ml-1m  # User activity & item popularity breakdowns
python part2/sparsity.py --data-dir ml-1m        # 100% / 50% / 20% sparsity ablation
python part2/plots.py                            # Generate summary PNG figures from saved CSVs
```

---

### 5.1 Part 1 Hyperparameter Tuning (Validation Set)

Neighborhood size $k$ and similarity metrics for Item-based kNN were tuned strictly on the **Validation set** ($k \in \{10, 20, 50, 100\}$ and Cosine vs. Adjusted Cosine).

To reproduce the validation tuning grid and generate `part1/outputs/tuning_results.csv`, run:
```bash
python part1/tune.py --data-dir ml-1m
```

#### Tuning Summary:
| Similarity | $k$ | Precision@10 | Recall@10 | NDCG@10 | Catalogue Coverage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cosine** | **100** | **0.0551** | **0.0915** | **0.0831** | **31.68%** |
| Cosine | 20 | 0.0533 | 0.0902 | 0.0816 | 35.80% |
| Cosine | 10 | 0.0530 | 0.0906 | 0.0812 | 36.21% |
| Cosine | 50 | 0.0533 | 0.0887 | 0.0807 | 34.05% |
| Adjusted Cosine | 20 | 0.0549 | 0.0716 | 0.0764 | 37.96% |
| Adjusted Cosine | 10 | 0.0537 | 0.0725 | 0.0762 | 41.02% |
| Adjusted Cosine | 50 | 0.0554 | 0.0703 | 0.0758 | 36.90% |
| Adjusted Cosine | 100 | 0.0550 | 0.0691 | 0.0748 | 37.08% |

*Conclusion*: Cosine similarity with $k=100$ yielded the highest ranking accuracy (NDCG@10 = 0.0831) on the validation set and is configured as the active baseline in `run.py`.

---

### 5.2 Part 2: Models & Hyperparameter Tuning (Validation Set)

Four models spanning four distinct paradigms were tuned strictly on the **Validation set** using an equal budget of **8 configurations per model** (`part2/tune.py`, generating `part2/outputs/tuning_results_part2.csv`):

| Model | Paradigm | Self-Implemented | Winning Validation Config | Val NDCG@10 |
| :--- | :--- | :---: | :--- | :---: |
| **Mult-VAE** | Neural CF (`torch`) | No | `hidden_dim=600, latent_dim=200, dropout=0.3, beta_max=0.2, epochs=30, lr=0.001` | **0.0864** |
| **BPR-MF** | Ranking-Loss CF (`implicit`) | No | `factors=128, learning_rate=0.01, regularization=0.01, iterations=100` | 0.0678 |
| **MF-SGD** | Latent-Factor CF (`surprise`) | No | `n_factors=32, reg_all=0.02, lr_all=0.005, n_epochs=25` | 0.0331 |
| **Content-Based** | Content-Based (Cosine) | **Yes** | `use_tfidf=False, weight_mode='centered', year_weight=0.5` | 0.0121 |

*Conclusion*: Each winning configuration is saved as the default in `part2/models/` and evaluated on the Test set (with 5-seed 95% confidence intervals, paired t-tests, disaggregation, and sparsity ablation) via `python run.py --data-dir ml-1m --part 2`.
