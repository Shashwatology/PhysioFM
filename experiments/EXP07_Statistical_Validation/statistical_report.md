# EXP07: Statistical Validation Report

**Source Evaluation:** `experiments/EXP06C_FeatureNorm`

## 1. Bland-Altman Analysis
- **Mean Bias:** 0.260 BPM
- **Limits of Agreement (95%):** [-18.160, 18.680]

## 2. Effect Size (Cohen's d)
- **Cohen's d:** 0.028
  - *Interpretation:* Negligible effect size (Excellent alignment).

## 3. Statistical Significance Testing
- **Normality Test (Shapiro-Wilk):** p = 2.2282e-02 (Errors are Not Normal)
- **Selected Test:** Wilcoxon signed-rank test
- **Test Statistic:** 9.0000
- **P-Value:** 8.4375e-01
  - *Conclusion:* Fail to reject Null Hypothesis. There is no statistically significant difference between predictions and targets.
