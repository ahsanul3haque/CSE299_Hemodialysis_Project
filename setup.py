from setuptools import setup, find_packages

setup(
    name="hemodialysis_crash_prediction",
    version="1.0.0",
    description="Clinical pipeline for predicting Intradialytic Hypotension (IDH)",
    author="Track A (Data & Infrastructure) & Track B (Analytics)",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.2",
        "imbalanced-learn>=0.9.0",
        "lightgbm>=3.3.2",
        "xgboost>=1.5.0",
        "shap>=0.41.0",
        "matplotlib>=3.5.0",
        "joblib>=1.1.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Healthcare Industry",
    ],
)