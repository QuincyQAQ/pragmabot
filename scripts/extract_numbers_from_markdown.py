#!/usr/bin/env python3
"""Extract prompt token counts or VLM response times from a Markdown evaluation log.

Edit ``FILENAME`` below and toggle ``count_token`` in ``__main__`` to switch modes.
"""

import re
import os
from pathlib import Path

# --- Edit this to point at the evaluation log you want to parse ---
FILENAME = "example_eval"
INPUT_PATH = Path(__file__).parent / "output" / f"{FILENAME}.md"


def extract_token_counts(file_path, count_token: bool):
    """
    Reads a markdown file, extracts numbers following 'Number of prompt tokens:'
    or 'VLM Response Time:', and prints them in a table.
    """

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return
    print(f"Extracting from file: {file_path}\n")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

            # Regex Explanation:
            # Number of prompt tokens:  -> Matches the literal phrase
            # \s* -> Matches any amount of whitespace (spaces/tabs)
            # ([\d,]+)                  -> Capturing group: Matches digits and commas (e.g., 1,000 or 1000)
            if count_token:
                pattern = r"Number of prompt tokens:\s*([\d,]+)"
            else:
                pattern = r"VLM Response Time:\s*([\d\.]+)\s*seconds"

            # Find all matches
            matches = re.findall(pattern, content)

            if not matches:
                print("No match found in the file.")
                return

            # Print the Table
            print("-" * 28)

            for index, count in enumerate(matches, 1):
                # Clean the comma if you want pure numbers later, but keep as string for display
                print(f"{count:<15}")

            print("-" * 28)
            print(f"Total entries found: {len(matches)}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    extract_token_counts(INPUT_PATH, count_token=False)
