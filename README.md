# Comparative Analysis: Logistic Regression & Tree Architectures on Titanic Dataset

A first-principles comparative study evaluating parametric linear models against non-parametric tree architectures on the standard **Kaggle Titanic Benchmark Dataset** ($n_{\text{train}} = 891$, $n_{\text{test}} = 418$).

- 📄 [Read the writeup](Titanic_ML_Project.pdf) — full mathematical derivations, loss convexity proofs, and information gain properties.
- 📓 [Explore the notebook](titanic_classification.ipynb) — complete Python implementation, model training, feature importance, and test set predictions.

---

## 📊 Summary of Results

All six model configurations were evaluated on the $418$ unseen Kaggle test passengers:

| Model Family | Variant | Kaggle Test Accuracy | Predicted Survivors ($n=418$) |
| :--- | :--- | :---: | :---: |
| **Linear** | Logistic Regression (Unregularized MLE) | **0.77751** | 161 |
| **Linear** | Logistic Regression (Ridge $L_2$) | **0.77751** | 159 |
| **Linear** | Logistic Regression (Lasso $L_1$) | **0.77751** | 159 |
| **Linear** | Logistic Regression (Elastic Net) | **0.77751** | 159 |
| **Tree-based** | Decision Tree (Single) | 0.77511 | 152 |
| **Tree-based** | Random Forest (Ensemble) | **0.77751** | 163 |

---

## 🔑 Key Takeaways

1. **High Inter-Model Consensus (98.56%)**: Regularized linear models and the Random Forest ensemble converged to identical test performance, matching on $98.56\%$ of test predictions. This confirms that feature engineering captured the core non-linear survival boundaries.
2. **Primary Survival Drivers**: Model interpretation across families reveals three key mechanisms driving survival:
   * **Demographic Priority**: `WomanOrChild` and `Sex` indicators served as the strongest positive drivers.
   * **Socioeconomic Status**: `Pclass` and `Fare` consistently provided strong secondary signals.
   * **Penalty Indicators**: `Title_Mr` and `LargeFamily` (>4 members) acted as the primary negative predictors.
3. **Variance Reduction**: The unconstrained single Decision Tree overfit slightly ($85.30\%$ train accuracy) and dropped to $0.77511$ on test data. Random Forest constrained depth and averaged predictions across 100 trees, reducing variance and recovering top-tier test performance ($0.77751$).
