# Presentation Slide Guide (12-Slide Deck)
**Course**: Deep Learning (Semester 5)  
**Mini Project**: Predicting Millionaire Status from Behavioral Habits  
**Dataset**: `Millionaire_Habits_Dataset.csv` (100,000 individuals, 236 feature inputs)  
**Target Feature**: `millionaire_status` (0 = Non-Millionaire, 1 = Millionaire)  
**Code Notebook**: [`millionaire_habits_deep_learning.ipynb`](./millionaire_habits_deep_learning.ipynb)

---

### Slide Page 1: Identity of Each Group Member
- **Title**: Deep Learning Classification: Predicting Millionaire Status from Lifestyle & Behavioral Habits
- **Course**: Deep Learning Mini Project - Group Presentation
- **Group Members**:
  1. **Member 1 (Leader)**: [Name] - NIM: [12345678]
  2. **Member 2**: [Name] - NIM: [12345679]
  3. **Member 3**: [Name] - NIM: [12345680]
  4. **Member 4**: [Name] - NIM: [12345681]
  5. **Member 5 (Optional)**: [Name] - NIM: [12345682]
- **Date**: Saturday, 17 Oct 2026

---

### Slide Page 2: Jobdesc for Each Group Member
- **Data Engineering & Leakage Prevention**:
  - Audited 274 columns and identified 35 potential target-leakage columns (net worth, total assets, total debt).
  - Developed end-to-end ColumnTransformer pipeline (Median imputation, One-Hot Encoding, StandardScaler).
- **Neural Network Architecture Design**:
  - Designed and implemented 3 PyTorch models: Baseline MLP, Deep Regularized MLP (BatchNorm + Dropout), and Tabular ResNet (Skip-connections + GELU).
- **Learning Strategy & Optimization**:
  - Configured BCEWithLogitsLoss, AdamW vs SGD vs RMSprop optimizers, and Cosine Annealing learning rate schedules.
- **Experimental Evaluation & Validation**:
  - Evaluated multi-metric performance (Accuracy, Precision, Recall, F1, ROC-AUC), Stratified 5-Fold Cross-Validation, and Permutation Feature Importance.
- **Report & Presentation Synthesis**:
  - Structured the 12-slide presentation deck, exported 300 DPI visualizations, and prepared QA speaking points.

---

### Slide Page 3: Description of the Problem Case
- **Problem Statement**:
  - Can an individual's financial success and millionaire status be predicted solely from their **daily habits, career choices, education, mindset, and financial behaviors**?
- **Dataset**: `Millionaire_Habits_Dataset.csv` (100,000 synthetic individuals, 274 raw features).
- **Target Variable**: `millionaire_status` ($y \in \{0, 1\}$).
  - Class 0 (Non-Millionaire): $57.51\%$
  - Class 1 (Millionaire): $42.49\%$ (well-balanced distribution).
- **Critical Scientific Constraint (Zero Data Leakage)**:
  - If direct balance-sheet items (`net_worth`, `total_assets`, `stocks_value`, `checking_savings`) are included, the model achieves trivial $100\%$ accuracy because millionaire status is defined as $\text{net\_worth} \ge \$1,000,000$.
  - **Our Solution**: Strictly eliminated all 35 balance-sheet total columns, training strictly on 236 genuine behavioral and demographic predictors.

---

### Slide Page 4: Description of the Data, Splitting Methods & K-Fold
- **Feature Diversity (236 Clean Features)**:
  - Demographics (Age, Urban/Rural, Education level, Marital status)
  - Career & Work (Industry, Tenure, Work hours, Remote frequency)
  - Saving & Investing Habits (Savings rate, Emergency fund months, Investment frequency)
  - Time & Productivity (Sleep duration, Deep work hours, Screen time)
  - Psychology & Consistency (Delayed gratification, Discipline score, Habit tracking)
- **Data Preprocessing**:
  - Numerical features: Median Imputation + Z-score Standardization ($z = \frac{x - \mu}{\sigma}$).
  - Categorical features: Constant Imputation + One-Hot Encoding (`handle_unknown='ignore'`).
  - Total post-encoded feature dimension: $531$ inputs.
- **Data Splitting**:
  - **Stratified Split**: $80\%$ Train ($80,000$), $10\%$ Validation ($10,000$), $10\%$ Test ($10,000$).
  - **Stratified 5-Fold Cross-Validation**: Validated across 5 non-overlapping folds to confirm statistical robustness.

---

### Slide Page 5: Architecture of Neural Networks / Deep Learning
We engineered and benchmarked three distinct architectures in PyTorch:
1. **Model 1: Baseline MLP**:
   - $\text{Linear}(531 \to 128) \to \text{ReLU} \to \text{Linear}(128 \to 64) \to \text{ReLU} \to \text{Linear}(64 \to 1)$
   - $76,417$ parameters.
2. **Model 2: Deep Regularized MLP**:
   - 4 hidden layers: $531 \to 256 \to 128 \to 64 \to 32 \to 1$.
   - Includes **Batch Normalization** (`BatchNorm1d`) and tiered **Dropout** ($0.25, 0.20, 0.12$).
   - $179,361$ parameters.
3. **Model 3: Modern Tabular ResNet**:
   - Inspired by recent advances in tabular deep learning (Gorishniy et al., NeurIPS 2021).
   - Input stem: $\text{Linear}(531 \to 256) + \text{BatchNorm} + \text{GELU}$.
   - 3 Residual blocks with skip connections: $x_{l+1} = \text{GELU}(x_l + \mathcal{F}(x_l))$.
   - Enables deep gradient preservation without degradation.
   - $380,289$ parameters.

---

### Slide Page 6: Learning Strategy & Parameter Tuning
- **Loss Function**: Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`) for stable sigmoid gradient computation.
- **Optimizer Benchmark**:
  - **AdamW** (Decoupled Weight Decay): $\text{lr}=10^{-3}$, weight decay $= 10^{-4}$ (Superior convergence).
  - **SGD with Nesterov Momentum** ($0.9$): $\text{lr}=10^{-2}$.
  - **RMSprop**: $\text{lr}=10^{-3}$.
- **Learning Rate Schedule**:
  - **Cosine Annealing** (`CosineAnnealingLR`): Smooth decay from $\eta_{max}$ to $\eta_{min}$, allowing optimal convergence.
- **Hyperparameter Grid Search**:
  - Explored Learning Rates ($\eta \in \{10^{-4}, 10^{-3}, 5\times 10^{-3}\}$) and Dropout ($p \in \{0.10, 0.25, 0.40\}$).
  - Optimal configuration: AdamW with $\eta=10^{-3}$, Dropout $=0.20-0.25$, Batch Size $=512$.

---

### Slide Page 7: Results Experiments - Model Comparison Table & Confusion Matrix
- **Image Asset**: `slides_export/slide7_confusion_matrices.png`
- **Experimental Benchmark on Held-out Test Set ($N=10,000$)**:

| Model Architecture | Train Time (s) | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline MLP** | $10.54$ | **$96.37\%$** | $95.06\%$ | **$96.47\%$** | **$0.9576$** | **$0.9957$** |
| **Deep Regularized MLP** | $20.16$ | $95.99\%$ | $94.54\%$ | $96.12\%$ | $0.9532$ | $0.9947$ |
| **Tabular ResNet** | $39.84$ | $95.70\%$ | $94.54\%$ | $95.39\%$ | $0.9496$ | $0.9938$ |

- **Key Insight**:
  - All deep learning models achieved $>95.5\%$ accuracy and $>0.99$ ROC-AUC purely from behavioral habits!
  - Confusion matrix reveals low False Positive Rate ($<3.5\%$) and low False Negative Rate ($<4.5\%$).

---

### Slide Page 8: Results Experiments - Learning Curves & ROC / PR Curves
- **Image Assets**: 
  - `slides_export/slide8_learning_curves.png`
  - `slides_export/slide8_roc_pr_curves.png`
- **Loss Curves**:
  - Smooth, monotonic decrease in validation loss across 15 epochs.
  - Zero divergence between training and validation loss, confirming absence of overfitting.
- **ROC and PR Curves**:
  - Area Under the Curve (AUC) exceeds $0.993$ across all three architectures.
  - PR-AUC exceeds $0.992$, demonstrating high precision even at near-perfect recall.

---

### Slide Page 9: Results Experiments - Parameter Tuning & 5-Fold Validation
- **Image Asset**: `slides_export/slide9_ablation_study.png`
- **Optimizer Comparison**:
  - AdamW converged fastest with highest stability ($95.7\%$ test accuracy).
  - SGD required significantly higher learning rate tuning and slower initial convergence.
- **Stratified 5-Fold Cross-Validation (Mean $\pm$ Std)**:
  - **Accuracy**: $95.78\% \pm 0.14\%$
  - **F1-Score**: $0.9504 \pm 0.0018$
  - **ROC-AUC**: $0.9941 \pm 0.0006$
  - Narrow standard deviations ($<0.2\%$) prove exceptional generalization across all data subsets.

---

### Slide Page 10: Conclusion & Habit Insights
- **Image Asset**: `slides_export/slide10_feature_importance.png`
- **Key Findings**:
  1. **Behavior Drives Wealth**: Daily habits, career tenure, and saving/investing discipline predict millionaire status with over $95.7\%$ accuracy, proving wealth is strongly rooted in predictable behavioral patterns.
  2. **Top Influential Habits Identified by Permutation Importance**:
     - **Investment Rate & Consistency**: Regular automated equity/portfolio contributions.
     - **Savings Rate & Emergency Buffer**: High savings rate relative to living expenses.
     - **Career Tenure & Experience**: Cumulative years of compounding career growth.
     - **Delayed Gratification & Financial Literacy**: High financial discipline scores.
  3. **Architectural Takeaway**: Modern tabular neural networks with Batch Normalization and Cosine Annealing learning rate schedules yield highly competitive, stable classifiers for complex tabular data.

---

### Slide Page 11: Code Repository & Project Structure
- **Repository URL**: `https://github.com/[your-group]/millionaire-habits-deep-learning`
- **Deliverables**:
  - `millionaire_habits_deep_learning.ipynb` (Full executed code with inline outputs)
  - `slides_export/` (All 300 DPI presentation figures & CSV benchmark tables)
  - `README.md` & `validation_report.txt`
- **Environment**: Python 3.14, PyTorch 2.12.0, Scikit-Learn 1.9.0.

---

### Slide Page 12: References
1. Gorishniy, Y., Rubachev, A., Khrulkov, V., & Babenko, A. (2021). *Revisiting Deep Learning Models for Tabular Data*. Advances in Neural Information Processing Systems (NeurIPS 2021).
2. Arik, S. Ö., & Pfister, T. (2021). *TabNet: Attentive Interpretable Tabular Learning*. AAAI Conference on Human Computation and Crowdsourcing.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
4. Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. International Conference on Learning Representations (ICLR).
5. Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization (AdamW)*. ICLR 2019.
