# UNICEF CPE Project  

This repository supports the **UNICEF Country Program Evaluations (CPE)** for the **ECARO region**, providing scripts and resources for **data processing** and **AI-assisted report generation**.

## Setup Instructions  

### 1. Environment Setup  

Ensure a consistent development environment using Conda:

1. **Create the Conda environment**:  
  ```bash
  conda env create -f environment.yml
  ```
2. **Activate the environment**:
  ```bash
  conda activate unicef-cpe
  ```
3. **Install the project in editable mode**:
  ```bash
  pip install -e .
  ```

## 2. Data Preparation

Download the necessary data following the **README** instructions located in the respective subfolders within data/raw.


- `raw` – Raw data files

The local directory structure should look like this:

```
/data
  ├── raw
```
## 3. Set the AI MODEL.
This version supports both OpenAI models and LLama models.
  - **For OpenAI models**:
	  1. Create a .env file in the current directory.
	  2. Add your OpenAI API key:
      ```plaintext
      OPENAI_API_KEY=<YOUR_OPENAI_TOKEN>
      ```
  - **For LLama models**:
    1. Download Ollama from [ollama.com](https://ollama.com).
	  2. Follow the installation instructions.
	  3. Download the LLama model as per the provided guidelines.

## 3. OCR Text extraction (optional) 
If text extraction from scanned PDFs is needed, install Tesseract:
**Install Tesseract** :
- **On macOS**: `brew install tesseract`
- **On Ubuntu**: 
  - `sudo apt install tesseract-ocr`
  - `sudo apt install libtesseract-dev`
- **On Windows**: Download and install Tesseract-OCR. Installer for Windows for Tesseract 3.05, Tesseract 4 and Tesseract 5 are available from [Tesseract at UB Mannheim] (https://github.com/UB-Mannheim/tesseract/wiki)

Further information are available in the [official page](https://tesseract-ocr.github.io/tessdoc/Installation.html)

## 4. Run the Data Pipeline & Generate Reports
Follow the instructions in Generate Report.ipynb to execute the pipeline and generate AI-assisted reports.


