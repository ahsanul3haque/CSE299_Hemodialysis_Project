# Week 3: Clinical Safety Metrics and Baseline Theory

## 1. Metric Selection: Prioritizing Recall
For this hemodialysis crash detection project, we prioritize **Recall** over overall Precision or Accuracy. 
* **Definition:** Recall measures how many of the actual positive cases (crashes) the model correctly identified. 
* **Clinical Justification:** In a medical setting, a False Negative (missing a patient who is actually crashing) is critically dangerous. We are willing to accept a higher rate of False Positives (false alarms) to ensure we capture as many true crashes as possible.

## 2. Algorithm Selection: Logistic Regression
While we tested multiple algorithms, Logistic Regression serves as our primary interpretability baseline.
* **Mechanism:** It uses the Sigmoid function to map patient clinical data to a probability score between 0 and 1, classifying a crash if the probability exceeds our threshold (typically 0.5).
* **Interpretability:** Logistic regression provides mathematical coefficients (representing log-odds). This allows us to explicitly show doctors which specific clinical variables (like a drop in Blood Pressure) increase or decrease the risk of a crash, rather than operating as a "black box."

## 3. Handling Class Imbalance
Our dataset is heavily imbalanced (stable sessions vastly outnumber crashes). 
* **The Strategy:** We applied `class_weight='balanced'` to our baseline Logistic Regression and Random Forest models. 
* **The Result:** This forces the algorithm to mathematically penalize itself more heavily for missing a rare crash class, prioritizing clinical safety over raw accuracy. (Note: Gradient Boosting lacked this parameter, resulting in a poor Recall of 0.25 and proving the necessity of SMOTE implementation).

## 4. Evaluation via ROC-AUC
To evaluate the models across all possible classification thresholds, we use the **ROC Curve** (plotting True Positive Rate vs. False Positive Rate) and calculate the **AUC (Area Under the Curve)**. An AUC closer to 1.0 indicates our model has a strong ability to separate stable sessions from crashing sessions.

