# /data — Kaggle Dataset Folder

Place the following CSV files here before running `train_models.py`:

| File | Kaggle Dataset | Target Column |
|---|---|---|
| `diabetes.csv` | [Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | `Outcome` |
| `heart.csv` | [Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) or [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) | `target` |
| `indian_liver_patient.csv` | [Indian Liver Patient Records](https://www.kaggle.com/datasets/uciml/indian-liver-patient-records) | `Dataset` |
| `kidney_disease.csv` | [Chronic Kidney Disease](https://www.kaggle.com/datasets/mansoordaku/ckdisease) | `classification` |

---

## Dataset Notes

### Heart Disease (`heart.csv`)
The UCI Heart Disease dataset comes in several versions on Kaggle. The most common ones are:
- **Cleveland version** (303 rows, 14 columns): Most widely used
- **Combined version** (1025 rows, 14 columns): Includes Cleveland + Hungary + Switzerland + VA Long Beach

Both work with this project. The preprocessing script expects these columns:
```
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
```

If your CSV has different column names, the script will auto-detect and use all columns except `target` as features.

**Download options:**
1. [Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) — 303 rows
2. [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) — 1025 rows (recommended)

After downloading, rename the file to `heart.csv` and place it in this folder.
