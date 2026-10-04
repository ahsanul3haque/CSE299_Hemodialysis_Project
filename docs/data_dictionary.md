Threshold used to identify Intradialytic Hypotension (IDH): 

Systolic Blood Pressure measured each hour from hour 0 (before the start of dialysis) to hour 5. If any of the pressure measured in hour 2-5 is lower by at least 20mmHg than any of the pressure measured in hour 0 and 1 Hypotension Measure (HYM) as the difference of the systolic blood pressures, then hypotension exists.

$$
\text{HYM} = \max(\text{SBP}_{\text{Hour 0}}, \text{SBP}_{\text{Hour 1}}) - \min(\text{SBP}_{\text{Hours 2--5}})
$$

$$
\text{IDH} = \begin{cases} 
  1 & \text{if HYM} \ge 20 \text{ mmHg} \\ 
  0 & \text{otherwise} 
\end{cases}
$$

Modeling Constraint & Data Leakage Prevention:
- The Target (y): The ground truth label indicating whether Intradialytic Hypotension (IDH) occurred is calculated using the full session timeline (Hours 0 through 5). 
- The Features (X): To ensure this system functions as a genuine proactive early-warning tool, the predictive model is strictly constrained to processing only "Hour 0" (pre-dialysis) parameters.
- Constraint Rule: All clinical vitals and machine settings recorded from Hour 1 to Hour 5 are explicitly dropped from the training set to prevent data leakage and perfectly simulate the information available to a clinician right before a session begins.




References:
J. A. Gómez-Pulido, J. M. Gómez-Pulido, D. Rodríguez-Puyol, and M. Vargas-Lombardo, "Predicting the appearance of hypotension during hemodialysis sessions using machine learning classifiers," Int. J. Environ. Res. Public Health, vol. 18, no. 5, p. 2364, Feb. 2021, doi: 10.3390/ijerph18052364.
