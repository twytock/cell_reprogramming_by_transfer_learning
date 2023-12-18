
## Repository description
Source code for generating the results of "Cell reprograming design by transfer learning of functional transcriptional networks."

## System Requirements and Installation

Python (v. 3.9) can be used to reproduce the results of this paper. 
To run the associated codes, the packages indicated in the file `reprog_design.yml` must be installed. 
The required version of Python and the dependent packages can be downloaded by creating a virtual environment. 
See the instructions at [miniforge](https://github.com/conda-forge/miniforge/tree/main/) to create a lightweight virtual environment. 
After installing miniforge, create a new virtual environment using `mamba create env -f reprog_design.yml`. 
Activate the environment with `mamba activate reprog_design`.

## Repository Organization
This respository is organized into the following directories:

1. `data/` -- Contains the source data necessary for the analysis
2. `code/` -- Contains the python and bash scripts and their documentation
3. `output/` -- Collects the outputs of the code execution

Executing the code in the `code/` directory will write files to the `output/` directory.

The script at `code/DataDownloader.py` will download the required source files into the `data/` directory.

Documentation specific to each script is provided in `code/README.md`.

The code is designed to be executed at the base directory (`./`) of this repository, e.g., `python code/<script_name>.py`.
