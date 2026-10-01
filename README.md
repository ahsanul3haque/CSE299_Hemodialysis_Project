# Proactive Risk Assessment and Counterfactual Recommendation for Hemodialysis Patients
**Course:** CSE299: Junior Design Project

**Section:** 11

**Faculty Initial:** MUO

**Team Members:** Ahsanul Haque (ID: 2411830642) | Syed Amin Hossain (ID: 2414265042)

## Week 1: Data Architecture & Clinical Ground Truth

### The Dataset Architecture
We are utilizing a peer-reviewed, open-access dataset from Mendeley Data. The repository contains clinical information extracted from 98,015 hemodialysis sessions corresponding to 758 patients. The data tracks 22 clinical parameters (measured 5 times per session), alongside the patient's gender and age. The data is divided into 11 subsets that successively include an increasing number of valid parameters, ranging from 2 up to 24 parameters. Our ingestion pipeline utilizes these CSVs, specifically focusing on the most robust subsets containing fully valid (non-null) clinical measurements.

### Clinical Ground Truth Definition
To establish a medically sound target variable without relying on arbitrary assumptions, we defer to the clinical definition of Intradialytic Hypotension (IDH). The target categorical response column `D_Hipotension` is defined based on standard clinical guidelines, where an adverse event is flagged if the patient experiences:
1. A drop in systolic blood pressure (SBP) >= 20 mmHg, OR
2. A decrease in Mean Arterial Pressure (MAP) by 10 mmHg associated with clinical symptoms, OR
3. An absolute nadir SBP < 90 mmHg requiring medical intervention.

By utilizing this pre-labeled dataset, our early-warning system is trained on verified, real-world clinical interventions.

### References & Data Sources
* **Dataset:** Gomez-Pulido, Juan A. (2021), "Dialysis database: sessions with valid data of clinical parameters", Mendeley Data, V1. 
  * **Link:** [https://data.mendeley.com/datasets/7kmtsmsgfw/1](https://data.mendeley.com/datasets/7kmtsmsgfw/1)
* **Clinical Methodology:** Mendoza-Pitti, L., et al. (2022). "Machine-Learning Model to Predict the Intradialytic Hypotension Based on Clinical-Analytical Data."
