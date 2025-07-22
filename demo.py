from src.usVisaClassifier.constants import CONFIG_FILE_PATH
from src.usVisaClassifier.utils import read_yaml_file

# Read config
config = read_yaml_file("config.yaml")



# Access database name safely with default fallback
DATABASE_NAME = config['database']['name']

print(f"Database Name: {DATABASE_NAME}")
