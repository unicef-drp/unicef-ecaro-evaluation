# UNICEF ECARO CPE Analysis  
## Raw Data Retrieval Guide  

This document outlines the steps to manually retrieve raw data for the **UNICEF ECARO CPE Analysis**. The instructions are valid as of **December 2024** and apply to the **Donor Statement Analysis Cube**.  

These steps must be repeated for each **Fund Sub Category**, selecting one of the following:  
- **Other Resources - Emergency**  
- **Other Resources - Regular**  
- **Regular Resources**  

## Steps to Retrieve Data  

### 1. Access Insight Platform  
- Go to [UNICEF Insight](https://insight.unicef.org/) and log in.  

### 2. Navigate to the Contribution Management Reports  
- On the landing page, go to **"Contribution"** under **Management Reports**.  

### 3. Select the Donor Statement Analysis Cube  
- In the **Finance & Admin** section, locate the **"Donor Statement"** category.  
- Select **"Donor Statement Analysis Cube"**.  

### 4. Apply Filters  
- Set **Year Hierarchy** till 2024  .  
- Set **Fund Types** to **All**.  
- Set **Fund Sub Categories** to **one of the following**:  
  - **Other Resources - Emergency**  
  - **Other Resources - Regular**  
  - **Regular Resources**  
- Set **Region** to **'ECAR'**.  

### 5. Select Data Values  
- Choose the following value:  
  - **Allocation**  

### 6. Configure Row Selection  
- Select the following as rows:  
  - **Output**  
  - **Donor Name**  

### 7. Export the Data  
- Download the extracted dataset.  
- Rename the file based on the selected **Fund Sub Category**:  

  - If **Other Resources - Emergency**:  
    ```plaintext
    insight-finance-admin-donor-statement-cube-output-donor-name-allocation-other-resources-emergency.xlsx
    ```  
  - If **Other Resources - Regular**:  
    ```plaintext
    insight-finance-admin-donor-statement-cube-output-donor-name-allocation-other-resources-regular.xlsx
    ```  
  - If **Regular Resources**:  
    ```plaintext
    insight-finance-admin-donor-statement-cube-output-donor-name-allocation-regular-resources.xlsx
    ```  
- **Before exporting**, ensure that **Row Labels** appear on **line 6** of the dataset.  
- Store the file in the **current folder**.  

