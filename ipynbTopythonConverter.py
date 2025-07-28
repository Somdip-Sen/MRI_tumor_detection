import json
import re
import sys
import os

def convert_notebook_to_script(notebook_path, output_path=None):
    """
    Converts a Jupyter Notebook (.ipynb) to a Python script (.py),
    extracting all code cells and removing all comments.

    Args:
        notebook_path (str): The full path to the input Jupyter Notebook file.
        output_path (str, optional): The full path for the output Python script.
                                     If None, it saves the script in the same
                                     directory with the same name but a .py extension.
                                     Defaults to None.
    """
    # Validate input path
    if not os.path.exists(notebook_path):
        print(f"Error: Input file not found at '{notebook_path}'")
        return

    if not notebook_path.endswith('.ipynb'):
        print(f"Error: Input file '{notebook_path}' is not a Jupyter Notebook.")
        return

    # Determine the output path if not provided
    if output_path is None:
        base_name = os.path.splitext(notebook_path)[0]
        output_path = base_name + '.py'
    
    print(f"Input Notebook: {notebook_path}")
    print(f"Output Script:  {output_path}")

    try:
        # Open and load the notebook file
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        final_code_lines = []
        
        # Extract code from cells
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                # Join all lines in the cell's source code
                source_code = "".join(cell.get('source', []))
                
                # Use regex to find and remove all comments (#...)
                code_without_comments = re.sub(r'#.*', '', source_code)
                
                # Add the cleaned code to our list if it's not just whitespace
                if code_without_comments.strip():
                    final_code_lines.append(code_without_comments)

        # Join the code from all cells, separated by a couple of newlines
        # for readability, and remove excessive blank lines.
        full_script = "\n\n".join(final_code_lines)
        full_script = re.sub(r'\n\s*\n', '\n\n', full_script).strip()

        # Write the final script to the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_script)
            
        print("\nConversion successful!")
        print(f"Clean Python script saved to '{output_path}'")

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{notebook_path}'. The file might be corrupted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # --- How to use this script ---
    # 1. Save this script as a Python file (e.g., `converter.py`).
    # 2. Change the `INPUT_NOTEBOOK_PATH` variable below to the path of your .ipynb file.
    # 3. (Optional) Change `OUTPUT_SCRIPT_PATH` if you want a custom name/location.
    # 4. Run the script from your terminal: python converter.py

    # --- Configuration ---
    # REQUIRED: Specify the path to your notebook file here
    INPUT_NOTEBOOK_PATH = "/Users/somdipsen/ML_Lab/Projects/MRI_Tumour/Code.ipynb"
    
    # OPTIONAL: Specify the output path. If set to None, it will save
    # the .py file in the same directory as the notebook.
    OUTPUT_SCRIPT_PATH = None 

    # --- Execution ---
    if INPUT_NOTEBOOK_PATH == "path/to/your/notebook.ipynb":
         print("Please update the 'INPUT_NOTEBOOK_PATH' variable inside the script first.")
    else:
        convert_notebook_to_script(INPUT_NOTEBOOK_PATH, OUTPUT_SCRIPT_PATH)


