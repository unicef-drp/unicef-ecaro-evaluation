"""
Functions for processing raw data files.
"""

import os
import re
from pathlib import Path

import pandas as pd
from unicef_cpe.config import RAW_DATA_DIR, FIXED_DATA_DIR
from .utils import get_ecaro_countries_mapping, read_and_combine_sheets


def make_df_start_from_row_labels(df: pd.DataFrame, starting_label = "Row Labels"):
    df = df.copy()
    if df.columns[0] != starting_label:
        row_labels_index = df[df.iloc[:, 0] == starting_label].index[0]
        # Set that row as the new column names
        column_names = df.iloc[row_labels_index].apply(
            lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)
        )

        df.columns = column_names
        df = df.iloc[row_labels_index + 1:].reset_index(drop=True)
    return df

def clean_indicators(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    if verbose:
        print("Shape before:", df.shape)

    df.rename(lambda x: re.sub(r"\s+", "_", x.strip().lower()), axis=1, inplace=True)
    mask = df["result_area"].fillna("").str.match(r"\d+") & df["indicator_unit"].ne(
        "TEXT"
    )
    df = df.loc[mask].copy()
    df.dropna(axis=1, how="all", inplace=True)
    df.dropna(subset=["indicator_actual"], inplace=True)

    def to_float(x):
        try:
            return float(x)
        except:
            return None

    for column in ("baseline_year", "target_year"):
        df[column] = df[column].astype(int)
    for column in ("baseline_value", "target_value", "indicator_actual"):
        df[column] = df[column].apply(to_float)

    mapping = get_ecaro_countries_mapping(keys="code", values="iso")
    df["country"] = df["business_area"].str.split(" - ").str.get(-1).replace(mapping)

    to_keep = [
        "country",
        "indicator_code",
        "indicator",
        "indicator_category",
        "indicator_unit",
        "baseline_year",
        "baseline_value",
        "target_year",
        "target_value",
        "finalization_date",
        "indicator_actual",
    ]
    df = df.reindex(to_keep, axis=1)
    if verbose:
        print("Shape after:", df.shape)
    return df


def process_utilisation_data(
    file_path: str, skiprows: int = 5, skipfooter: int = 1, **kwargs
):
    mapping = get_ecaro_countries_mapping(keys="code", values="iso")
    df_funds = pd.read_excel(
        file_path, skiprows=skiprows, skipfooter=skipfooter, **kwargs
    )
    df_funds.columns = [
        "results_area",
        "other_resources_emergency",
        "other_resources_regular",
        "regular_resources",
        "grand_total",
    ]
    records = []
    country = None
    for record in df_funds.to_dict(orient="records"):
        match = re.search(
            r".+- \d{4}", record["results_area"]
        )  # e.g., 'Armenia - 0260', 'Bosnia and Herzegovina - 0530
        if match or "ECARO" in record["results_area"]:
            _, code = record["results_area"].split("-")
            country = mapping.get(code.strip(), record["results_area"])
            continue
        record["country"] = country
        records.append(record)
    df_funds = pd.DataFrame(records)
    return df_funds


def read_vision_programme_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path, skiprows=4)
    df = make_df_start_from_row_labels(df)
    file_name, _ = os.path.splitext(
        file_path.name
    )  # e.g. "VISION Programme Coding Analysis Cube – CP By Year.xlsx"
    *_, layout = file_name.lower().split(
        " - "
    )  # e.g. "VISION Programme Coding Analysis Cube – CP By Year"
    groups, columns = layout.split(" by ")  # e.g. "CP By Year"
    groups = groups.replace(" ", "_")
    columns = columns.replace(" ", "_")

    mapping = get_ecaro_countries_mapping(keys="code", values="iso")
    country_iso = None
    records = []
    for record in df.to_dict(orient="records"):
        if record["Row Labels"] == "Grand Total":
            continue
        elif not isinstance(record["Row Labels"], str):
            continue

        # e.g., 'Albania - 0090', 'North Macedonia - 2660'
        match = re.match(r"[A-Z]\w+.+(\d{4})$", record["Row Labels"])
        if match:
            country_code = match.groups()[0]
            country_iso = mapping[country_code]
            continue

        record["country"] = country_iso
        record.pop("Grand Total")
        record[groups] = record.pop("Row Labels")
        records.append(record)

    df = pd.DataFrame(records)
    df = df.melt(id_vars=["country", groups], var_name=columns, value_name="value")
    if columns == "year":
        df["year"] = df["year"].astype(int)
    df.dropna(subset=["value"], ignore_index=True, inplace=True)
    return df


def read_vision_programme_data_gl(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path, skiprows=5)
    mapping = get_ecaro_countries_mapping(keys="code", values="iso")
    country_iso = None
    records = []
    for record in df.to_dict(orient="records"):
        if record["Row Labels"] == "Grand Total":
            continue
        elif not isinstance(record["Row Labels"], str):
            continue

        # e.g., 'Albania - 0090', 'North Macedonia - 2660'
        match = re.match(r"[A-Z]\w+.+(\d{4})$", record["Row Labels"])
        if match:
            country_code = match.groups()[0]
            country_iso = mapping[country_code]
            continue

        # e.g., 'CONTRACTUAL SERVICES', 'TRAVEL
        match = re.match(r"^[A-Z]+\b", record["Row Labels"])
        if match:
            cost_category = record["Row Labels"]
            continue

        # e.g., not "0007000110 Programme Evaluation Services", so "Other Resources - Emergency"
        match = re.match(r"^\d+", record["Row Labels"])
        if not match:
            source = record["Row Labels"]
            continue

        record["country"] = country_iso
        record["source"] = source
        record["cost_category"] = cost_category
        record.pop("Grand Total")
        record["gl_account"] = record.pop("Row Labels")
        records.append(record)

    df = pd.DataFrame(records)
    df = df.melt(
        id_vars=["country", "source", "cost_category", "gl_account"],
        var_name="year",
        value_name="value",
    )
    df["year"] = df["year"].astype(int)
    df.dropna(subset=["value"], ignore_index=True, inplace=True)
    return df


def read_vision_programme_data_activities(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path, skiprows=5)
    df = make_df_start_from_row_labels(df)
    mapping = get_ecaro_countries_mapping(keys="code", values="iso")
    source = None
    records = []
    for record in df.to_dict(orient="records"):
        if record["Row Labels"] == "Grand Total":
            continue
        elif not isinstance(record["Row Labels"], str):
            continue

        # e.g., "0090/A0/04/800/889/003 OTHER CROSS-SECTORAL ACTIVITIES"
        match = re.match(r"^(\d+)", record["Row Labels"])
        if not match:
            source = record["Row Labels"]
            continue
        elif match.group() not in mapping:
            continue

        record["country"] = mapping[match.group()]
        record["source"] = source
        record.pop("Grand Total")
        record["activity"] = record.pop("Row Labels")
        records.append(record)

    df = pd.DataFrame(records)
    df = df.melt(
        id_vars=["country", "source", "activity"], var_name="year", value_name="value"
    )
    df["year"] = df["year"].astype(int)
    df.dropna(subset=["value"], ignore_index=True, inplace=True)
    return df


def get_top_outputs(df_funds: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    # convert to wide format
    df_funds = df_funds.pivot_table(
        index=["country", "cp_wbs", "cp", "output"],
        columns="year",
        values="value",
        aggfunc="sum",
    )
    df_funds.reset_index(inplace=True)
    df_funds.columns.name = None
    # calculate the total over the years
    df_funds["total"] = df_funds.filter(regex=r"\d{4}", axis=1).sum(axis=1)

    # arrange the outputs by CP and total amount
    df_funds["cp_year"] = df_funds["cp"].str.extract(r"(\d{4})")
    df_funds.dropna(subset="cp_year", inplace=True)
    df_funds["cp_year"] = df_funds["cp_year"].astype(int)
    df_funds.sort_values(
        ["cp_year", "total"], ascending=[True, False], ignore_index=True, inplace=True
    )

    # get outputs by the highest total and group the rest into "other"
    df_top = df_funds.groupby(["country", "cp_wbs", "cp"]).head(n - 1)
    df_other = (
        df_funds.drop(df_top.index)
        .groupby(["country", "cp_wbs", "cp"], as_index=False)
        .sum()
    )
    df_other["output"] = "OTHER OUTPUTS"

    # join the data and clean up the dataframe
    df = pd.concat([df_top, df_other], axis=0)
    df.drop(["total", "cp_year"], axis=1, inplace=True)
    df = df.melt(
        id_vars=["country", "cp_wbs", "cp", "output"],
        var_name="year",
        value_name="value",
    )
    df["year"] = df["year"].astype(int)
    return df



def get_area_mapping(mapping_by='goal_area'):
    # Define the mapping path
    mapping_path = FIXED_DATA_DIR / 'PIDB SIC Mapped.xlsx'

    # Define the sheets and corresponding CP values for goal areas and result areas
    sheets_goal_areas = [('Goal Area 2018-2021', '2018-2021'), ('Goal Area 2022-2025', '2022-2025')]
    sheets_result_areas = [('Result Area 2018-2021', '2018-2021'), ('Result Area 2022-2025', '2022-2025')]
    if mapping_by=='goal_area':
        # Read and combine goal area data
        map_area_df = read_and_combine_sheets(sheets_goal_areas, mapping_path)
        mapping_dict = map_area_df.set_index('Goal Area Code')['Goal Area'].to_dict()
    elif mapping_by=='result_area':
        # Read and combine result area data
        map_area_df = read_and_combine_sheets(sheets_result_areas, mapping_path)
        mapping_dict = map_area_df.set_index('Result Area Code')['Result Area'].to_dict()
    
    return mapping_dict

    
def get_strategy_mapping(mapping_by='gic_name'):
    # Define the mapping path
    mapping_path = FIXED_DATA_DIR / 'GIC - Generic Intervention Codes.xlsx'
    map_area_df = pd.read_excel(mapping_path, sheet_name='CP GIC List')
    map_area_df = map_area_df.astype(str)
    
    # Strip whitespace from all string columns
    map_area_df = map_area_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    if mapping_by=='gic_name':
        mapping_dict = map_area_df.set_index('GIC Code')['GIC Name'].to_dict()
    elif mapping_by=='strategy_name':
        mapping_dict = map_area_df.set_index('GIC Code')['Implementing strategy Description'].to_dict()

    elif mapping_by=='strategy_code':
        mapping_dict = map_area_df.set_index('GIC Code')['Implementing strategy Code'].to_dict()
    
    elif mapping_by=='strategy_mapped':
        map_area_df = pd.read_excel(mapping_path, sheet_name='Strategy Map')
        map_area_df = map_area_df.astype(str)
        mapping_dict = map_area_df.set_index('Implementing strategy Description CP-2018-2021')['Implementing strategy Description CP-2022-2025'].to_dict()

    return mapping_dict
    



def get_programme_structure():
    data_path = RAW_DATA_DIR / 'insight-programme-programme-structure/Programme Structure.csv'
    df = pd.read_csv(data_path)

    def standardize_programme_column(column_name):
        column_name = column_name.lower()
        column_name = column_name.replace('cp_','country_programme_')
        column_name = column_name.replace('pcr_','outcome_')
        column_name = column_name.replace('intermediate_result_','output_')
        column_name = column_name.replace('ir_','output_')
        column_name = column_name.replace('full_activity_','activity_')
        column_name = column_name.replace('_statement','_description')
        column_name = column_name.replace('_full_text','_description')
        column_name = column_name.replace('_wbs','_code')
        column_name = column_name.replace('_cd_name','_name')
        column_name = column_name.replace('_cd','_code')
        return column_name

    #Standardize columns
    df.columns = [standardize_programme_column(column_name) for column_name in df.columns]

    df = df.astype(str)
    
    # Strip whitespace from all string columns
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    #Covert utilized to float
    df['utilized'] = df['utilized'].str.replace(',','')
    df['utilized'] = df['utilized'].astype(float)

    def get_result_area_code(specific_intervention_code):
        # Extract the first two parts of the code, if available
        result_area_code = specific_intervention_code
        if isinstance(specific_intervention_code, str) and '-' in specific_intervention_code:
            result_area_code = '-'.join(specific_intervention_code.split('-')[0:2])
        return result_area_code

    def get_goal_area_code(specific_intervention_code):
        # Extract only the first part of the code
        goal_area_code = specific_intervention_code
        if isinstance(specific_intervention_code, str) and '-' in specific_intervention_code:
            goal_area_code = specific_intervention_code.split('-')[0]
        return goal_area_code

    # Create new columns with the 'country' and 'business_area_name' information from 'business_area' column
    df['country'] = df['business_area'].apply(lambda x: x.split(' - ')[0] if isinstance(x, str) else None)
    df['business_area_name'] = df['business_area'].apply(lambda x: x.split(' - ')[0] if isinstance(x, str) else None)

    # Apply the functions to extract goal area and result area codes
    df['goal_area_code'] = df['specific_intervention_code'].apply(get_goal_area_code)
    df['result_area_code'] = df['specific_intervention_code'].apply(get_result_area_code)

    df['goal_area'] = df['goal_area_code'].replace(get_area_mapping(mapping_by='goal_area'))
    df['result_area'] = df['result_area_code'].replace(get_area_mapping(mapping_by='result_area'))

    df['generic_intervention_name'] = df['generic_intervention_code'].replace(get_strategy_mapping(mapping_by='gic_name')).fillna('Unknown')
    df['strategy_name'] = df['generic_intervention_code'].replace(get_strategy_mapping(mapping_by='strategy_name')).fillna('Unknown')
    df['strategy_code'] = df['generic_intervention_code'].replace(get_strategy_mapping(mapping_by='strategy_code')).fillna('Unknown')
        
    df['strategy_mapped'] = df['strategy_name'].replace(get_strategy_mapping(mapping_by='strategy_mapped')).fillna('Unknown')   

    # Apply the function to the DataFrame and expand the result into two new columns
    df[['cp_start_year', 'cp_end_year', 'cp']] = df.apply(cp_start_end_year, axis=1, result_type='expand')

    return df


def cp_start_end_year(row, column_name='country_programme_name'):
    # Regular expression to extract two years
    year_regex = re.compile(r'(\d{4})\D+(\d{4})')
    
    cp_name = row[column_name]
        
    match = year_regex.search(cp_name)
    
    if match:
        # If a match is found, extract start and end years
        start_year, end_year = match.groups()
        return int(start_year), int(end_year), f'CP ({start_year}-{end_year})'
    else:
        # If no match is found, return None for both years
        return None, None, None


def read_partner_types_data() -> pd.DataFrame:
    data_path = RAW_DATA_DIR / 'insight-ram3-partner-types' / 'ecar-partner-list.xlsx'
    df = pd.read_excel(data_path, sheet_name='by BA, Partner & FR', skiprows=6)
    df.columns = [col.lower().replace(' ','_') for col in df.columns]
    df = df.astype(str)
    df = df[df.iloc[:, 0] != 'Grand Total']

    return df