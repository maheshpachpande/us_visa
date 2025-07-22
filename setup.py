import os
from setuptools import setup, find_packages

# Package metadata
__version__ = "0.0.0"
PROJECT_NAME = "usVisaClassifier"
AUTHOR = "Mahesh Pachpande"
AUTHOR_EMAIL = "pachpandemahesh300@gmail.com"



def get_requirements_list(filename="requirements.txt"):
    """
    Reads a requirements file and returns a list of packages.
    
    Args:
        filename (str): Path to the requirements file.
        
    Returns:
        List[str]: A list of requirement strings.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found.")
    
    with open(filename, "r") as f:
        requirements = f.read().splitlines()
        requirements = [r.strip() for r in requirements if r.strip() and r != "-e ."]
    return requirements

# Setup configuration
setup(
    name=PROJECT_NAME,
    version=__version__,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=get_requirements_list(),  
)
