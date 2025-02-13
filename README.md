# UNICEF CPE Project

This repository is part of the UNICEF Country Program Evaluations (CPE) project. It provides scripts, notebooks, and resources necessary for processing data and generating AI-assisted reports for various countries in the ECARO region.

## Setup Instructions

### 1. Create and Activate Conda Environment

Set up a consistent development environment using Conda. This ensures all dependencies are correctly installed:

  a. Create the environment: Use the environment.yml file to create a Conda environment named unicef-cpe, with all specified dependencies pre-configured.

  ```
  conda env create --name unicef-cpe -f environment.yml
  ```

  b. Activate the environment: Switch to the newly created environment to start working with its dependencies.

  ```
  conda activate unicef-cpe
  ```

  c.	Install the project in editable mode: Link the local source code to the environment, allowing for live updates during development.

  ```
  pip install -e .
  ```

### 2. Download SpaCy Language Model

Install the required language model for natural language processing (NLP):

```bash
python -m spacy download en_core_web_sm
```

### 3. Data Preparation

Download the necessary data from [Nextcloud UNICEF CPE](https://nextcloud.rowsquared.com/index.php/f/593844). You will need to copy the following directories into the local `data/` folder:

- `raw` – Raw data files
- `output` – Processed output files

The local directory structure should look like this:

```
/data
  ├── raw
  └── output
```

#### 4. OCR Text extraction (optional)
**Install Tesseract** :
- **On macOS**: `brew install tesseract`
- **On Ubuntu**: 
  - `sudo apt install tesseract-ocr`
  - `sudo apt install libtesseract-dev`
- **On Windows**: Download and install Tesseract-OCR. Installer for Windows for Tesseract 3.05, Tesseract 4 and Tesseract 5 are available from [Tesseract at UB Mannheim] (https://github.com/UB-Mannheim/tesseract/wiki)

Further information are available in the [official page](https://tesseract-ocr.github.io/tessdoc/Installation.html)

## Notebook Guidelines

- All notebooks for data processing must be stored in the `/pipelines` directory.
- Notebooks should be named with an incremental numeric prefix (e.g., `01_preprocessing.ipynb`, `02_analysis.ipynb`).
- At the beginning of each notebook, include a Markdown section specifying which **Evaluation Questions (EQs)** the notebook addresses.
  
  Example:
  ```markdown
  # Relevance-1
  ## To what extent UNICEF positioning in the country and implementation strategies enable itself to respond to those needs?
  ```

- Each notebook in `/pipelines` should output one or more Excel files to the `/data/output` directory. Ensure that the Excel files include **all countries** in a single file (i.e., no separate files per country).

## Generating HTML Reports

To generate AI-assisted HTML reports for each country, navigate to the `/reports/notebook` directory and run the following commands in the terminal:

```bash
quarto render report.qmd -o unicef-ecaro-cpe-report-ARM-v$VERSION.html -P COUNTRY=ARM -M subtitle="Country Report for Armenia"

quarto render report.qmd -o unicef-ecaro-cpe-report-AZE-v$VERSION.html -P COUNTRY=AZE -M subtitle="Country Report for Azerbaijan"

quarto render report.qmd -o unicef-ecaro-cpe-report-GEO-v$VERSION.html -P COUNTRY=GEO -M subtitle="Country Report for Georgia"

quarto render report.qmd -o unicef-ecaro-cpe-report-KAZ-v$VERSION.html -P COUNTRY=KAZ -M subtitle="Country Report for Kazakhstan"

quarto render report.qmd -o unicef-ecaro-cpe-report-MKD-v$VERSION.html -P COUNTRY=MKD -M subtitle="Country Report for North Macedonia"

quarto render report.qmd -o unicef-ecaro-cpe-report-BIH-v$VERSION.html -P COUNTRY=BIH -M subtitle="Country Report for Bosnia and Herzegovina"

quarto render report.qmd -o unicef-ecaro-cpe-report-BLR-v$VERSION.html -P COUNTRY=BLR -M subtitle="Country Report for Belarus"
```

Replace `$VERSION` with the appropriate report version date (e.g., `24-08-31`, `24-09-06`).

## Folder Structure

Below is a recommended folder structure for this project:

```
/unicef-cpe
  ├── /data            # Contains raw and processed data
  │   ├── raw
  │   └── output
  ├── /pipelines       # Notebooks for data processing
  ├── /reports         # Report generation files
  │   └── /notebook    # Quarto notebooks for report generation
  ├── environment.yml  # Conda environment setup
  └── README.md        # Project documentation
```
