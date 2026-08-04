# Predicting Diamond Prices with Regression Models
**CSC 310 Final Project**

A machine learning project that builds and compares five regression models to predict diamond prices in USD from the classic ggplot2 diamonds dataset.

---

## Problem Statement

A diamond's price is determined by its size and quality grades — the "4 Cs": carat, cut, color, and clarity — along with physical measurements. This project answers three questions:

1. Can we accurately predict diamond price from physical and quality attributes?
2. Which attributes matter most?
3. Does a non-linear model beat a simple linear one?

---

## Dataset

**Source:** [ggplot2 diamonds dataset](https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/diamonds.html)  
**Size:** 53,940 diamonds, 10 features

| Feature | Type | Description |
|---|---|---|
| `carat` | Numeric | Weight of the diamond (1 carat = 0.2g) |
| `cut` | Ordinal | Cut quality: Fair → Good → Very Good → Premium → Ideal |
| `color` | Ordinal | Color grade: J (worst) → D (best) |
| `clarity` | Ordinal | Clarity grade: I1 (worst) → IF (best) |
| `depth` | Numeric | Total depth percentage |
| `table` | Numeric | Width of top relative to widest point (%) |
| `x`, `y`, `z` | Numeric | Physical dimensions in mm |
| `price` | Numeric | **Target** — price in USD ($326–$18,823) |

---

## Key Findings

- **Size dominates price.** Carat and physical dimensions (x, y, z) account for the vast majority of a diamond's price.
- **Non-linear models win.** Linear regression tops out at ~89% R², while the Random Forest reaches ~98% — an 8-point gap that's statistically significant.
- **Quality is a size-adjusted correction.** Raw correlations suggest better grades are *cheaper* (counterintuitive), but this is a confounding effect: high-grade diamonds tend to be small. Once the model controls for size, better clarity and color correctly *increase* the predicted price — as confirmed by SHAP values.
- **Depth, table, and cut barely matter** for price prediction once size and quality grades are accounted for.

---

## Models Compared

| Model | CV R² | Test R² | Test MAE |
|---|---|---|---|
| **Random Forest** *(tuned)* | **0.981** | **0.983** | **$266** |
| Decision Tree *(tuned)* | 0.976 | 0.978 | $310 |
| KNN *(tuned)* | 0.969 | 0.971 | $354 |
| Lasso Regression | 0.902 | — | — |
| Linear Regression | 0.892 | 0.907 | $802 |

The **Random Forest** is the best model. Its 95% confidence interval (0.9793, 0.9825) does not overlap with the Decision Tree's (0.9733, 0.9776), and a paired t-test confirms the difference is statistically significant (p < 0.001).

---

## Project Structure

```
CSC310-Final-Project/
├── CSC310_Final_Project_Diamonds.ipynb   # Full analysis notebook
└── README.md
```

---

## How to Run

**Requirements:** Python 3.8+, Jupyter Notebook or JupyterLab

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy shap
jupyter notebook CSC310_Final_Project_Diamonds.ipynb
```

The notebook downloads the diamonds dataset automatically on first run — no manual download needed.

---

## Notebook Outline

1. **Introduction & Data Loading** — dataset overview, summary statistics
2. **Data Cleaning** — removed 20 rows with zero dimensions, 145 duplicates
3. **Exploratory Data Analysis** — price distribution, feature distributions, boxplots, scatter matrix, correlation heatmap
4. **Preprocessing** — ordinal encoding for cut/color/clarity, train/test split (80/20), StandardScaler in pipelines
5. **Five Models** — Linear Regression, Lasso, KNN, Decision Tree, Random Forest (all tuned with GridSearchCV)
6. **Model Comparison** — 95% confidence intervals, paired t-test
7. **Overfitting Check** — train vs. test R² gap analysis
8. **Model Interpretation** — decision tree visualization, feature importances, SHAP beeswarm and waterfall plots
9. **Bias Check** — residuals by price quartile
10. **Conclusion**

---

## Libraries Used

`numpy` · `pandas` · `matplotlib` · `seaborn` · `scikit-learn` · `scipy` · `shap`
