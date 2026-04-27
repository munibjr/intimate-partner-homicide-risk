from setuptools import setup, find_packages

setup(
    name="intimate-partner-homicide-risk",
    version="0.1.0",
    description="ML system for identifying escalation patterns in partner violence cases",
    author="Munibjr",
    author_email="munib.080@gmail.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "shap",
        "matplotlib",
    ],
)
