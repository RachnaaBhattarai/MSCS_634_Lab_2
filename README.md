# MSCS 634 Lab 2: Classification Using KNN and RNN Algorithms

## Purpose

This lab explores the performance of K-Nearest Neighbors (KNN) and Radius Neighbors (RNN) classifiers using the Wine Dataset from the `sklearn` library. The primary objective is to evaluate how variations in parameters—`k` for KNN and `radius` for RNN—affect classification accuracy, and to understand practical considerations for selecting between these algorithms.

---

## Key Insights

### KNN Results
- **Accuracy Trends:** KNN achieved consistent accuracy (~80.56%) for `k` values of 5, 11, 15, and 21. A lower accuracy (77.78%) was noted at `k=1`.
- **Observation:** Accuracy plateaued after `k=5`, suggesting that a small neighborhood suffices for this dataset.

### RNN Results
- **Accuracy Trends:** RNN performance varied with radius values:
  - Highest accuracy (100%) was achieved at `radius=0.5` with scaled data.
  - Performance dropped significantly for other radii (e.g., 0.2 and 2.0).
- **Observation:** RNN is sensitive to the radius parameter and requires proper scaling for optimal performance.

### Comparison
- **Best Models:**
  - **KNN:** `k=5`, Accuracy = 80.56%
  - **RNN:** `radius=0.5`, Accuracy = 100% (with scaled data)
- **Takeaway:** RNN outperformed KNN in this experiment but required more careful parameter tuning and preprocessing. KNN was more robust with minimal setup.

---

## Challenges and Decisions

### Data Scaling for RNN
- RNN initially underperformed on unscaled data.
- Applying `MinMaxScaler` corrected this, emphasizing the need for scaling in distance-based models.

### Parameter Selection
- **KNN:** Multiple `k` values were tested to determine the optimal neighborhood size.
- **RNN:** Initial radius values (350–600) were ineffective; tuning to smaller radii (0.2–2.0) after scaling improved results.

### Outlier Handling
- Used `outlier_label='most_frequent'` in RNN to handle cases with no neighbors, preventing runtime errors.

---

## When to Use KNN vs. RNN

### Prefer KNN When:
- The data distribution is relatively uniform.
- Simplicity and model interpretability are desired.
- Feature scaling is not feasible (although it still helps).

### Prefer RNN When:
- Data density varies across the feature space.
- A meaningful radius can be chosen based on domain knowledge.
- Feature scaling is applied, and adaptive neighborhood sizes are needed.

---

## Conclusion

This lab highlighted the trade-offs between KNN and RNN classifiers. KNN offers simplicity and consistency, making it ideal for general use. In contrast, RNN provides greater flexibility and performance when scaling and fine-tuning are appropriately handled. Proper preprocessing and careful parameter selection are crucial for achieving optimal results with either approach.
