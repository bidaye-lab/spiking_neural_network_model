import subprocess
from pathlib import Path

def create_notebooks(script_folder, notebook_folder):
    '''Create Jupyter notebooks from Python scripts

    For each .py file in script_folder, create a .ipynb file in with the
    same name in notebook_folder.

    Parameters
    ----------
    script_folder : path-like
        Path to folder with .py files
    notebook_folder : path-like
        Path to folder where .ipynb files will be saved
    '''

    script_files = Path(script_folder).glob("*.py")
    notebook_folder = Path(notebook_folder)
    notebook_folder.mkdir(exist_ok=True, parents=True)

    for script_file in script_files:
        notebook_file = notebook_folder / (script_file.stem + ".ipynb")
        command = f'jupytext --to ipynb {script_file} --output {notebook_file}'
        print(command)
        subprocess.run(command.split())


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description="Create notebooks from script files using jupytext",
        epilog='IMPORTANT: does not handle white spaces in file names')
    
    parser.add_argument("-s", "--script-folder", type=str, default="scripts")
    parser.add_argument("-n", "--notebook-folder", type=str, default="notebooks")
    args = parser.parse_args()

    create_notebooks(args.script_folder, args.notebook_folder)