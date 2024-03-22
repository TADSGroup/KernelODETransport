.PHONY: all permissions setup check-environment data-download clean


ENV_NAME="kode_env"

# Default target
all: permissions setup check-environment data-download test-data clean

## Make setup.sh executable
permissions:
	@echo "Making setup.sh file executable..."
	chmod +x setup.sh

## Setup Python environment
setup:
	@echo "Setting up Conda environment..."
	./setup.sh


## Check Python environment
check-environment:
	@echo "Activating the Conda environment $(ENV_NAME) and running tests..."
	@bash -c "source activate $(ENV_NAME); python -m pytest tests/test_environment.py"


## Download datasets
data-download:
	@echo "Downloading the datasets"
	@mkdir -p data
	@curl -o data/data.tar.gz "https://zenodo.org/records/1161203/files/data.tar.gz?download=1"
	@tar -xzvf data/data.tar.gz --strip-components=1 -C data/


## Test that the data downloads are correct
test-data:
	@echo "Checking that the downloaded data can be correctly loaded..."
	@bash -c "source activate $(ENV_NAME); python -m pytest tests/test_data.py"


## Delete all compiled Python files
 clean:
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@echo "Cleaning compiled Python files"
