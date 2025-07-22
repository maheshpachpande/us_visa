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