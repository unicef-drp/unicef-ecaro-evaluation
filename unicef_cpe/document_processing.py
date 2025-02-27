import re
from collections import defaultdict
import pandas as pd


def calculate_average_line_length(text):
    lines = text.split("\n")
    total_length = sum(len(line.strip()) for line in lines if len(line.strip()) > 0)
    num_lines = len([line for line in lines if len(line.strip()) > 0])
    return total_length / num_lines if num_lines > 0 else 0


def starts_with_capital_after_numeric(stripped_line: str) -> bool:
    # Regular expression to find the first alphabetic character after any leading digits and spaces
    match = re.search(r'\b[a-zA-Z]', stripped_line)
    # Check if the matched character is uppercase
    return bool(match and match.group().isupper())

def remove_empty_lines(text):
    # Split text by lines, filter out empty lines, and join the remaining lines
    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(non_empty_lines)

def remove_leading_number(text):
    # Regular expression to match a number at the beginning followed by whitespace
    return re.sub(r"^\d+\.?\s*", "", text)


def extract_section_titles(text, threshold=50, base_min_lines_between_titles=3, start_with_digit=False):
        
    # Split the text by lines
    lines = remove_empty_lines(text).split("\n")
    
    # Initialize an empty list to store section titles and the previous title's line number
    section_titles = []
    cleaned_titles = []
    last_title_index = 0  # Start before the beginning of the document
    
    # Regex pattern for titles: either start with digits followed by a space/dot or start directly with letters. In lateer one it should be Capital
    if start_with_digit:
        digit_pattern = re.compile(r"^(\d+(\.\d+)*[ .]+)+[A-Z][a-zA-Z]{2,}")
    else:
        digit_pattern = re.compile(r"^((\d+(\.\d+)*[ .]+)*)[A-Z][a-zA-Z]{2,}")

    # Iterate through the lines, looking for shorter lines that are surrounded by longer ones
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        
        # Check if it should start with a digit based on the `start_with_digit` parameter
        starts_with_digit_condition = bool(digit_pattern.match(stripped_line)) 
        # Check the rules for a valid section title
        if (
            len(stripped_line) > 0                                    # Line is not empty
            and len(stripped_line) < threshold                        # Line is shorter than threshold
            and not stripped_line.endswith(('.', ';', ',', '-')) # Does not end with punctuation
            and i - last_title_index > base_min_lines_between_titles  # At least adjusted number of lines from the last title
            and starts_with_digit_condition                           # Condition for starting with a digit if required
        ):
            # Check if the next line is longer (indicating that this is likely a title) or empty
            if True:#i + 1 < len(lines) and len(lines[i + 1].strip()) > threshold:
                cleaned_title = remove_leading_number(stripped_line) 
                if cleaned_title not in cleaned_titles:
                    section_titles.append(stripped_line)
                    cleaned_titles.append(cleaned_title)
                    last_title_index = i

    return section_titles


def extract_section_titles_multiple_documents(documents):
    # Create a defaultdict to store the frequency of each title across documents
    title_frequency = defaultdict(int)
    
    # Process each document and extract section titles
    for doc in documents:
        titles = extract_section_titles(doc)
        for title in titles:
            title_frequency[title] += 1
    
    return title_frequency


def filter_common_titles(title_frequency, min_frequency):
    # Filter titles that appear at least 'min_frequency' times
    common_titles = [title for title, freq in title_frequency.items() if freq >= min_frequency]
    return common_titles

def extract_keyword_chunks(text, keyword, token_window=100):
    """
    Extracts chunks of text centered around a keyword, with a given number of words before and after.
    
    Parameters:
    - text (str): The full text to search within.
    - keyword (str): The keyword to find in the text.
    - token_window (int): Number of words before and after the keyword to include in the chunk.
    
    Returns:
    - List of text chunks containing the keyword within the specified token range.
    """
    # Split text into words (tokens) using regex to handle punctuation properly
    tokens = re.findall(r'\b\w+\b', text)
    
    # Find all occurrences of the keyword (case insensitive)
    keyword_indices = [i for i, token in enumerate(tokens) if token.lower() == keyword.lower()]
    title = ""
    # Extract chunks centered around the keyword
    chunks = []
    for index in keyword_indices:
        start = max(index - token_window, 0)
        end = min(index + token_window + 1, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append([title,chunk])

    return chunks


def extract_sections_with_text(text, section_titles):
    """
    Extracts section titles and their corresponding text by splitting the document based on the section titles.
    Returns a list of tuples: (section_title, section_text).
    """
    # Split the text by new lines
    lines = remove_empty_lines(text).split("\n")
    
    sections = []
    current_section = None
    current_text = ""

    for line in lines:
        stripped_line = line.strip()
        
        if stripped_line in section_titles:  # If the line is a section title
            if current_section:  # Save the current section and its text
                sections.append((current_section, current_text.strip()))
            # Start a new section
            current_section = stripped_line
            current_text = ""
        else:
            # Append the line to the current section text
            current_text += line + " "
    
    # Add the last section if it exists
    if current_section:
        sections.append((current_section, current_text.strip()))
    
    return sections


