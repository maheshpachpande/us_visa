# us_visa
A comprehensive MLOps pipeline that automates model training, validation, deployment, and monitoring with CI/CD integration, feature stores, experiment tracking, and real-time performance monitoring to enable scalable, reliable machine learning operations in production environments.

### Step:1 Create a GitHub repository to manage code versioning effectively.
- git clone <https link>
- Git is the tool, GitHub is the online service that helps you use that tool for collaboration and sharing (code versioning).


### Step:2 Create a Conda virtual environment to isolate dependencies and manage packages efficiently for your project.
- conda create -n visa python=3.9 -y
- conda activate visa
- conda env remove -n visa -y -y


### Step:3 

- "PyPI (Python Package Index) is the official repository for Python packages, allowing developers to publish and install reusable code libraries to streamline development and promote modularity."

- A well-organized project *FOLDER STRUCTURE* with packages is essential to ensure modularity, scalability, and maintainability of code in a production-ready machine learning pipeline.

- template.py (python template.py)

### Step:4

- setup.py makes a Python project installable, reusable, and ready for distribution or deployment by defining its metadata and dependencies.

- setup.py (python setup.py install)

- requirements.txt (pip install -r requirements.txt)
- The -r flag is required to indicate you want pip to install from a requirements file, not a single package by that name. It helps automate dependency management and maintain consistency across environments.

-  What Does -e . Mean?
    -- e stands for editable.
    -- . refers to the current directory (where setup.py is located).


### Step:5

- Craete Exception file


### Step:6
# =======================================================================================================
"""
DataIngestion class is a modular and production-ready data ingestion component of an ML pipeline, 
built specifically for a US Visa classification project. 
Its primary responsibility is to extract raw data from MongoDB, 
store it locally (feature store), and split it into training and testing sets.
"""
# =======================================================================================================

"""🎯 PURPOSE:
This `DataValidation` class automates validation checks on ingested datasets before they enter the ML training pipeline.

📦 KEY RESPONSIBILITIES:
1. Validate schema integrity (number of columns and required columns)
2. Detect missing values and drift in numerical/categorical columns
3. Generate and save a drift report using the `Evidently` library
4. Create and persist a `DataValidationArtifact` for pipeline traceability
"""

# =======================================================================================================

""" This `DataTransformation` class is part of an ML pipeline, specifically the Data Transformation stage. Its primary roles are:

Load raw data from the Data Ingestion stage.

Preprocess features: encoding, scaling, power transformation.

Feature engineering: add derived features like company_age.

Balance data using SMOTEENN (combines SMOTE oversampling + ENN cleaning).

Save transformed datasets and preprocessor object for model training."""

# =======================================================================================================

"""
Model Trainer class is a component of an ML pipeline, specifically the Model Training stage. Its primary responsibility is to train multiple ML models using hyperparameter tuning (GridSearchCV), select the best model, evaluate it using metrics (F1, Precision, Recall), and save both the model and metrics as artifacts in a structured pipeline.
"""