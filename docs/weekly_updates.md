# CSE299 Weekly Updates

## Week 1
**Date:** October 1, 2026
**Status:** Complete

**Update Pitch:**
For Week 1, our primary goal was establishing a medically defensible foundation and our software architecture. We successfully sourced a 2021 peer-reviewed clinical dataset from Mendeley Data containing over 97,000 real hemodialysis sessions. Rather than using synthetic data or guessing, we documented the exact clinical blood-pressure thresholds the medical researchers used to define intradialytic hypotension, ensuring our ground truth is medically sound.

On the engineering side, we initialized our GitHub repository, configured our data privacy settings, and wrote a Python ingestion pipeline using Pandas. Running this script validated the schema and gave us three critical insights for next week: our target variable has a moderate 67-to-33 class imbalance, the clinical features are labeled in Spanish, and the temperature column is 100% empty. We are now perfectly positioned to build our data cleaning and preprocessing pipeline for Week 2."