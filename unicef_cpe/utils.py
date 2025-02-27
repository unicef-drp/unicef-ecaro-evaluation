"""
Miscellaneous utility functions.
"""
import glob
import json
from importlib import resources
from textwrap import wrap
from typing import Literal
from pathlib import Path
import re
import os
import pandas as pd

FIELD = Literal["iso", "name", "code"]


def get_ecaro_countries(field: FIELD = "name", priority: bool = False) -> list[str]:
    """
    Get a list of programme countries for ECARO.

    Parameters
    ----------
    field : FIELD, default='name'
        Name of a country field.
    priority : bool, default=False
        If True, only return priority countries for CPEs.

    Returns
    -------
    countries : dict[str, str]
        List of countries either as names or iso or codes.
    """
    files = resources.files("data.fixed")
    with files.joinpath("countries.json").open() as file:
        countries = json.load(file)
    countries = [
        country[field] for country in countries if country["priority"] or not priority
    ]
    return countries


def get_ecaro_countries_mapping(
    keys: FIELD = "iso", values: FIELD = "name", priority: bool = False
) -> dict[str, str]:
    """
    Get a mapping of programme countries for ECARO.

    Parameters
    ----------
    keys : FIELD, default='iso'
        Name of a country field to be used as dictionary keys.
    values : FIELD, default='name'
        Name of a country field to be used as dictionary values.
    priority : bool, default=False
        If True, only return priority countries for CPEs.

    Returns
    -------
    countries : dict[str, str]
        Mapping of countries.
    """
    keys = get_ecaro_countries(field=keys, priority=priority)
    values = get_ecaro_countries(field=values, priority=priority)
    mapping = dict(zip(keys, values))
    return mapping


def get_pidb_entries(
    level: Literal["domain", "area", "sic"],
    kind: Literal["name", "code", "both"],
    year: int = 2018,
) -> list[str]:
    """
    Get a list of entries from the Programme Information Database (PIDB).

    Parameters
    ----------
    level : Literal['domain', 'area', 'sic']
        One of the three levels of available in PIDB where 'domain'
        is the most generic and 'sic' is the most specific.
    kind : Literal['name', 'code', 'both']
        Kind of information to return, i.e., names, codes or names with codes ('both').
    year : int, default=2018
        Year of the PIDB version.

    Returns
    -------
    entries : list[str]
        Unique PIDB entries of the given level. The order of items is the same for all `kind` values.
    """
    files = resources.files("data.fixed")
    with files.joinpath(f"unicef-pidb-{year}.json").open() as file:
        pidb = json.load(file)
    if kind == "name" or kind == "code":
        entries = [record[f"{level}_{kind}"] for record in pidb]
    elif kind == "both":
        entries = [
            record[f"{level}_code"] + " - " + record[f"{level}_name"] for record in pidb
        ]
    else:
        raise ValueError(f"Unknown `kind` value {kind}.")
    entries = list(
        dict.fromkeys(entries)
    )  # get a list of unique elements preserving the order
    return entries


def wrap_text_with_br(text: str, width: int = 40) -> str:
    return "<br>".join(wrap(text, width=width))


def replace_business_areas_with_iso(
    business_area: str, ignore_missing: bool = False
) -> str:
    """
    Examples
    --------
    >>> replace_business_areas_with_iso("Albania - 0090")
    "ALB"
    """
    business_area_code = business_area.split("-")[-1].strip()
    mapping = get_ecaro_countries_mapping(keys="code", values="iso")
    iso = mapping.get(business_area_code)
    if iso is None and not ignore_missing:
        raise ValueError(f"Could not map {business_area_code} code.")
    return iso


def write_sheet_to_excel(dataframe: pd.DataFrame, file_path: Path, sheet_name: str = 'main') -> None:
    """Write a DataFrame to an Excel sheet.

    If the file does not exist, create it and write the DataFrame to the sheet.
    If the file exists, append the DataFrame to the sheet.
    """
    if file_path.suffix != '.xlsx':
        file_path = file_path.with_suffix('.xlsx')
    if not file_path.exists():
        with pd.ExcelWriter(path=file_path, engine='openpyxl', mode='w') as writer:
            pd.DataFrame().to_excel(writer, sheet_name='README', index=False)
    
    with pd.ExcelWriter(path=file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        dataframe.to_excel(writer, sheet_name=sheet_name, index=False)

def format_number(n):
    if n >= 1_000_000:
        return f'{n / 1_000_000:.1f}M'
    elif n >= 1_000:
        return f'{n / 1_000:.1f}K'
    else:
        return str(n)
    

def read_corpora_jsonl(file_path: Path, typ: Literal['rdletters', 'coars', 'sitans', 'cpds', 'roars', 'psns'] ) -> pd.DataFrame:
    with open(file_path, 'r') as rf:
        df = pd.read_json(rf, lines=True)
    typ_section_map = {
        'rdletters': 'Regional Director Letters',
        'coars': 'Country Office Annual Reports',
        'sitans': 'Situation Analyses',
        'cpds': 'Country Programme Documents',
        'roars': 'Regional Office Annual Reports',
        'psns': 'Programme Strategy Notes',
    }
    df['section'] = df['file_type'].map(typ_section_map)
    return (df
            .query(f"file_type == '{typ}'")
            .sort_values(by=['country', 'year'], ignore_index=True))


def read_excel_sheet(path: Path, sheet_name: str):
    """
    Reads a specific sheet from an Excel file.
    Args:
        path (Path): The file path to the Excel file.
        sheet_name (str): The name of the sheet to read.
    Returns:
        DataFrame: A pandas DataFrame containing the data from the specified sheet.
    """

    
    return pd.read_excel(path, sheet_name=sheet_name)


def split_long_text(text: str, max_length: int = 30000, overlap: int = 0) -> list[str]:
    """
    Splits a long text into smaller chunks. Tries to split at the nearest newline character or period. Overlap not guaranteed
    Args:
        text (str): The text to split.
        max_length (int): The maximum length of each chunk.
    Returns:
        List[str]: A list of smaller text chunks.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    while len(text) > max_length:
        # Find the best split index based on newline or period
        split_idx_newline = text.rfind('\n', 1, max_length)
        split_idx_period = text.rfind('. ', 1, max_length)
        split_idx = max(split_idx_newline, split_idx_period)
        
        # If no suitable split index is found, split at max_length
        if split_idx == -1:
            split_idx = max_length
        
        # Find the best overlap index based on newline or period
        overlap_idx_newline = text.rfind('\n', 1, max_length + overlap)
        overlap_idx_period = text.rfind('. ', 1, max_length + overlap)
        overlap_idx = max(overlap_idx_newline, overlap_idx_period)
        
        # If no suitable overlap index is found, use max_length + overlap
        if overlap_idx == -1:
            overlap_idx = max_length + overlap
        
        # Append the chunk and update the text
        chunks.append(text[:overlap_idx])
        text = text[split_idx:]
    
    # Append the remaining text as the last chunk
    chunks.append(text)
    return chunks


def remove_unprintable_chars(s: str):
    """
    Clean a string by removing NULL bytes and control characters.
    """
    if isinstance(s, str):
        return ''.join(c for c in s if c.isprintable())
    return s


def sanitize_dataframe(df: pd.DataFrame):
    """
    Clean a DataFrame by removing NULL bytes and control characters.
    """
    return df.applymap(remove_unprintable_chars)


def read_and_combine_sheets(sheet_info, mapping_path):
    df_list = []
    for sheet_name, cp in sheet_info:
        # Read the Excel sheet and add the CP column
        df = pd.read_excel(mapping_path, sheet_name=sheet_name)
        df['CP'] = cp
        # Convert all columns to string type
        df = df.astype(str)
        
        # Strip whitespace from all string columns
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df_list.append(df)
    
    # Concatenate all the DataFrames into one
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


def beautify_labels(label):
    new_label = label.replace('_name','').replace('_code','')
    new_label = new_label.replace('_',' ').title()
    return new_label


def clean_text(text):
    if isinstance(text, str) is False:
        return text  # Return NaN as is
    # Remove '\n', '\r', and other special characters
    cleaned_text = re.sub(r'[\r\n]+', ' ', text)  # Replaces \n and \r with a space
    # Remove any other non-alphanumeric characters (except spaces and punctuation if desired)
    cleaned_text = re.sub(r'[^\w\s.,:;?!-]', '', cleaned_text)  # Keeps letters, numbers, spaces, and punctuation like . and ,
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    # Strip leading/trailing spaces
    return cleaned_text.strip()

def remove_section_title(markdown_text: str, h_title='###') -> str:
    """
    This function removes the section title in Markdown format, specifically lines starting with '###'.
    """
    lines = markdown_text.splitlines()
    
    # Filter out any line that starts with '###'
    filtered_lines = [line for line in lines if not line.strip().startswith(h_title)]
    
    # Join the filtered lines back into a single string
    result = "\n".join(filtered_lines).strip()
    
    return result


def generate_output_excel(proj_root, country):
    # Find all processed Excel files dynamically
    data_path = proj_root / f"data/processed/{country}/"
    output_path = proj_root / f"data/outputs/{country}/cpe_evaluation_data.xlsx"
    processed_files = glob.glob(os.path.join(data_path, "*.xlsx"))

    if not processed_files:
        raise ValueError(f"No processed Excel files found in {data_path}")


    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create missing directories

    # Create a new Excel writer
    with pd.ExcelWriter(output_path) as writer:
        for file in processed_files:
            try:
                # Extract sheet name from filename
                sheet_name = os.path.splitext(os.path.basename(file))[0]
                
                # Read the Excel file (handle multiple sheets if necessary)
                excel_data = pd.ExcelFile(file)
                for sheet in excel_data.sheet_names:
                    df = excel_data.parse(sheet)
                    writer.sheets[f"{sheet_name}"] = df
                    df.to_excel(writer, sheet_name=f"{sheet_name}", index=False)
            except Exception as e:
                print(f"Error processing {file}: {e}")

    print(f"Final combined output saved as {output_path}")

