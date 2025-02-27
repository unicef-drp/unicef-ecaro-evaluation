# UNICEF ECARO CPE Analysis  
## Raw Data Retrieval Guide  

This document outlines the steps to manually retrieve raw data for the **UNICEF ECARO CPE Analysis**. The instructions are valid as of **December 2024** and apply to the **Partnership Analysis Cube**.  

## Steps to Retrieve Data  

### 1. Access the Insight Platform  
- Go to [UNICEF Insight](https://insight.unicef.org/) and log in.  

### 2. Select the Partnership Analysis Cube  
- In the **DCT & Partners** section, navigate to the **Cubes** category.  
- Select **"Partnership Analysis Cube"**.  

### 3. Apply Filters  
- **Implementing Business Area Hierarchy** → Set to **ECARO**.  
- **Posting Date Hierarchy** → Set from **2016 to 2025**.  

### 4. Configure Row Selection  
Select the following fields as rows:  
- **FR Document Number**  
- **FR Start Year**  
- **FR End Year**  
- **Donor Code**  
- **Donor Name**  
- **Fund Category**  
- **Fund Sub-Category**  
- **Fund Type**  
- **Grant**  

### 5. Export the Data  
- Download the extracted dataset.  
- Rename the file to:  
  ```plaintext
  insight-dct-partners-funds-info.xlsx
  ```
- **Before exporting**, ensure that the **FR Document Number** appears on **line 7** of the dataset.
- Store the file in the current folder for further processing.