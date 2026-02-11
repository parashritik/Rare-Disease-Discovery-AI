"""
Helper functions for cleaning datasets in the Rare Disease Drug Repurposing project.
"""

import pandas as pd


def normalize_text_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Convert text in a column to lowercase and strip leading/trailing spaces.

    Useful for standardizing disease names, drug names, or any text field before
    matching or merging. Modifies the DataFrame in place and returns it for chaining.

    Args:
        df: The DataFrame to modify.
        column: Name of the column containing text to normalize.

    Returns:
        The same DataFrame with the column normalized.
    """
    if column not in df.columns:
        return df
    # Only normalize non-missing values; leave NaN as-is so fill_missing_with_placeholder can handle them
    mask = df[column].notna()
    df.loc[mask, column] = df.loc[mask, column].astype(str).str.strip().str.lower()
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.

    Keeps the first occurrence of each duplicate row and drops the rest.
    Use after cleaning so that identical records (e.g. same disease-gene pair)
    appear only once.

    Args:
        df: The DataFrame from which to remove duplicates.

    Returns:
        A new DataFrame with duplicate rows removed.
    """
    return df.drop_duplicates()


def fill_missing_with_placeholder(
    df: pd.DataFrame, column: str, placeholder: str = "unknown"
) -> pd.DataFrame:
    """
    Fill missing values in a column with a placeholder string.

    Replaces NaN, None, and empty strings with the placeholder so downstream
    code does not have to handle missing values (e.g. for display or export).

    Args:
        df: The DataFrame to modify.
        column: Name of the column in which to fill missing values.
        placeholder: Value to use for missing entries (default: "unknown").

    Returns:
        The same DataFrame with missing values in that column filled.
    """
    if column not in df.columns:
        return df
    # Fill NaN/None with placeholder
    df[column] = df[column].fillna(placeholder)
    # Optionally replace empty strings if they represent "missing"
    df.loc[df[column].astype(str).str.strip() == "", column] = placeholder
    return df
