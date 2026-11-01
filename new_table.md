| Model                    | Stage                                    |   Threshold |   Accuracy |   Precision |   Recall |   F1_Score |   ROC_AUC |   PR_AUC |
|:-------------------------|:-----------------------------------------|------------:|-----------:|------------:|---------:|-----------:|----------:|---------:|
| Random Forest (Weighted) | 1. Baseline (Default @ 0.50)             |        0.5  |     0.8273 |      0.588  |   0.5108 |     0.5467 |    0.8123 |   0.5801 |
| Random Forest (Weighted) | 2. Tuned (Default @ 0.50)                |        0.5  |     0.8134 |      0.5375 |   0.6061 |     0.5697 |    0.8205 |   0.5991 |
| Random Forest (Weighted) | 3. Tuned + Max-F1 Thresh (0.49)          |        0.49 |     0.8079 |      0.524  |   0.6277 |     0.5712 |    0.8205 |   0.5991 |
| Random Forest (Weighted) | 4. Tuned + Clinical Safety Thresh (0.47) |        0.47 |     0.7984 |      0.5042 |   0.6529 |     0.569  |    0.8205 |   0.5991 |
| Random Forest (SMOTE)    | 1. Baseline (Default @ 0.50)             |        0.5  |     0.8233 |      0.5758 |   0.5054 |     0.5383 |    0.8102 |   0.5667 |
| Random Forest (SMOTE)    | 2. Tuned (Default @ 0.50)                |        0.5  |     0.8185 |      0.553  |   0.5719 |     0.5623 |    0.8124 |   0.5761 |
| Random Forest (SMOTE)    | 3. Tuned + Max-F1 Thresh (0.48)          |        0.48 |     0.8138 |      0.539  |   0.5971 |     0.5666 |    0.8124 |   0.5761 |
| Random Forest (SMOTE)    | 4. Tuned + Clinical Safety Thresh (0.42) |        0.42 |     0.7771 |      0.468  |   0.6835 |     0.5556 |    0.8124 |   0.5761 |
| Random Forest (ADASYN)   | 1. Baseline (Default @ 0.50)             |        0.5  |     0.8167 |      0.5547 |   0.5108 |     0.5318 |    0.7996 |   0.5496 |
| Random Forest (ADASYN)   | 2. Tuned (Default @ 0.50)                |        0.5  |     0.805  |      0.5208 |   0.5414 |     0.5309 |    0.8077 |   0.5568 |
| Random Forest (ADASYN)   | 3. Tuned + Max-F1 Thresh (0.44)          |        0.44 |     0.7804 |      0.4723 |   0.6583 |     0.55   |    0.8077 |   0.5568 |
| Random Forest (ADASYN)   | 4. Tuned + Clinical Safety Thresh (0.44) |        0.44 |     0.7804 |      0.4723 |   0.6583 |     0.55   |    0.8077 |   0.5568 |
| XGBoost (Weighted)       | 1. Baseline (Default @ 0.50)             |        0.5  |     0.7936 |      0.4943 |   0.5414 |     0.5167 |    0.7894 |   0.5432 |
| XGBoost (Weighted)       | 2. Tuned (Default @ 0.50)                |        0.5  |     0.7874 |      0.4838 |   0.6439 |     0.5525 |    0.8156 |   0.599  |
| XGBoost (Weighted)       | 3. Tuned + Max-F1 Thresh (0.54)          |        0.54 |     0.8105 |      0.5307 |   0.6061 |     0.5659 |    0.8156 |   0.599  |
| XGBoost (Weighted)       | 4. Tuned + Clinical Safety Thresh (0.46) |        0.46 |     0.7698 |      0.4579 |   0.705  |     0.5552 |    0.8156 |   0.599  |
| XGBoost (ADASYN)         | 1. Baseline (Default @ 0.50)             |        0.5  |     0.8178 |      0.584  |   0.3687 |     0.452  |    0.7795 |   0.5255 |
| XGBoost (ADASYN)         | 2. Tuned (Default @ 0.50)                |        0.5  |     0.7522 |      0.4277 |   0.6385 |     0.5123 |    0.7892 |   0.5216 |
| XGBoost (ADASYN)         | 3. Tuned + Max-F1 Thresh (0.48)          |        0.48 |     0.7262 |      0.403  |   0.714  |     0.5152 |    0.7892 |   0.5216 |
| XGBoost (ADASYN)         | 4. Tuned + Clinical Safety Thresh (0.48) |        0.48 |     0.7262 |      0.403  |   0.714  |     0.5152 |    0.7892 |   0.5216 |
| LightGBM (Weighted)      | 1. Baseline (Default @ 0.50)             |        0.5  |     0.7764 |      0.4661 |   0.6673 |     0.5488 |    0.8124 |   0.5912 |
| LightGBM (Weighted)      | 2. Tuned (Default @ 0.50)                |        0.5  |     0.7823 |      0.4751 |   0.6529 |     0.55   |    0.8169 |   0.5871 |
| LightGBM (Weighted)      | 3. Tuned + Max-F1 Thresh (0.53)          |        0.53 |     0.7984 |      0.5043 |   0.6313 |     0.5607 |    0.8169 |   0.5871 |
| LightGBM (Weighted)      | 4. Tuned + Clinical Safety Thresh (0.47) |        0.47 |     0.7702 |      0.4577 |   0.6906 |     0.5505 |    0.8169 |   0.5871 |
| LightGBM (ADASYN)        | 1. Baseline (Default @ 0.50)             |        0.5  |     0.8354 |      0.6551 |   0.4065 |     0.5017 |    0.8049 |   0.5721 |
| LightGBM (ADASYN)        | 2. Tuned (Default @ 0.50)                |        0.5  |     0.7911 |      0.4875 |   0.491  |     0.4892 |    0.7794 |   0.5136 |
| LightGBM (ADASYN)        | 3. Tuned + Max-F1 Thresh (0.44)          |        0.44 |     0.7496 |      0.4245 |   0.6421 |     0.5111 |    0.7794 |   0.5136 |
| LightGBM (ADASYN)        | 4. Tuned + Clinical Safety Thresh (0.43) |        0.43 |     0.7397 |      0.4137 |   0.6637 |     0.5097 |    0.7794 |   0.5136 |