# Recommender Systems Project - Group 3

---

## 1. Repository Structure

```text
recommender_systems_project/
├── README.md                              # Reproducibility instructions & architecture
├── requirements.txt                       # Explicit pinned dependencies
├── run.py                                 # Single documented command entry point
├── Recommender_Systems_Results_Template.xlsm # Standardized results record workbook
├── data/                                  # Local dataset folder (ignored from Git)
│   └── ml-1m/
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
```

---

## 2. Environment & Requirements

- **Programming Language**: Python 3.12+ (tested on Python 3.12 / 3.14 via WSL Ubuntu 24.04).
- **Core Libraries**: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `tqdm`, `openpyxl`.

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
# Run only Part 1
python run.py --data-dir ml-1m --part 1
```

---

## 5. Hyperparameter Tuning (Validation Set)

Neighborhood size $k$ and similarity metrics for Item-based kNN were tuned strictly on the **Validation set** ($k \in \{10, 20, 50, 100\}$ and Cosine vs. Adjusted Cosine).

To reproduce the validation tuning grid and generate `part1/outputs/tuning_results.csv`, run:
```bash
python part1/tune.py --data-dir ml-1m
```

### Tuning Summary:
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
