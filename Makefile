.PHONY: setup data-download clean all



## Make setup.sh executable
permissions:
	echo "Making setup.sh file executable..."
	chmod +x setup.sh

## Setup Python environment
setup:
	echo "Setting up Conda environment..."
	./setup.sh


## Download datasets
data-download:
	mkdir -p data
	wget -P data/power.zip "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
	unzip -o data/power.zip -d data/ &&  rm data/power.zip

## Delete all compiled Python files
 clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


# Default target
all: setup data-download clean
