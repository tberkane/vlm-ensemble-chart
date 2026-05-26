import csv
import io
import re


def normalize_table_value(value: str) -> str:
    """Normalize a table cell value to:
    - Lowercase all text
    - Remove all non-numeric chars except . in between two digits (decimal point)
      or - just before a digit (minus sign)
    - Convert all empty values to nan
    - Handle numbers with comma separators (e.g. 1,310,000 -> 1310000)
    - Handle scientific notation (e.g. 4.1e+01 -> 41.0)
    """
    if not value or not value.strip():
        return 0

    value = value.lower().strip()

    if not value:
        return 0

    # Remove commas in numbers for easier extraction, but not inside decimal part
    # This will ensure numbers like "1,310,000" are turned into "1310000"
    value_without_commas = re.sub(r"(?<=\d),(?=\d)", "", value)

    # Check for scientific notation pattern (e.g., 4.1e+01, 4.1e-01, 4.1e01)
    scientific_pattern = r"-?\d+(?:\.\d+)?[e][+-]?\d+"
    scientific_match = re.search(scientific_pattern, value_without_commas)

    if scientific_match:
        try:
            # Convert scientific notation to float, then to string
            num_value = float(scientific_match.group())
            # Format as decimal number, ensuring at least one decimal place
            if num_value % 1 == 0:
                return str(int(num_value)) + ".0"
            else:
                return str(num_value)
        except ValueError:
            pass

    # Now, use regex to extract valid numeric patterns:
    # - Optional minus sign at the start
    # - One or more digits (possibly grouped after removing commas)
    # - Optional decimal point followed by digits
    # This pattern matches: -123, 123, 123.45, -123.45, 1310000, etc.
    pattern = r"-?\d+(?:\.\d+)?"
    matches = re.findall(pattern, value_without_commas)

    if matches:
        return matches[0]
    else:
        return 0


def normalize_tsv_table(tsv_content: str) -> str:
    """Normalize a TSV table string, processing all values except first row and first column.

    Args:
        tsv_content: TSV table as a string

    Returns:
        Normalized TSV table as a string
    """
    # Parse TSV
    reader = csv.reader(io.StringIO(tsv_content), delimiter="\t")
    rows = list(reader)

    if not rows:
        return tsv_content

    # Process each cell (skip first row and first column)
    processed_rows = []
    for row_idx, row in enumerate(rows):
        processed_row = []
        for col_idx, cell in enumerate(row):
            if row_idx == 0 or col_idx == 0:
                # Keep first row and first column as-is
                processed_row.append(cell)
            else:
                # Process the cell value
                processed_row.append(normalize_table_value(cell))
        processed_rows.append(processed_row)

    # Convert back to TSV string
    output = io.StringIO()
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    for row in processed_rows:
        writer.writerow(row)
    return output.getvalue()
