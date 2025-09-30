from setuptools import setup, find_packages

setup(
    name="obesity_detection",           
    version="0.1.0",                   
    author=" NIKHIL NAIK",                    
    author_email="pnikhilkumarnaik@example.com",
    description="A machine learning project for obesity detection",
    packages=find_packages(),           
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "imbalanced-learn",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "joblib",
        "streamlit"
    ],
    python_requires='>=3.8',              
)
