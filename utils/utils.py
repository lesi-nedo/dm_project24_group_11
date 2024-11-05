import shutil
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import importlib.util

def move_to_tmp(file_path, keep_original=False):
    """
    Check if file exists and move it to /tmp folder.
    
    Args:
        file_path (str): Path to the file to be moved
        keep_original (bool): If True, copy file instead of moving it
    
    Returns:
        str: Path to the new location if successful, None if file doesn't exist
    """
    try:
        # Convert to Path object for easier handling
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            print(f"File {file_path} does not exist")
            return None
            
        # Create a new filename with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        tmp_path = Path('/tmp') / new_filename
        
        # Move or copy the file
        if keep_original:
            shutil.copy2(file_path, tmp_path)
            print(f"File copied to {tmp_path}")
        else:
            shutil.move(file_path, tmp_path)
            print(f"File moved to {tmp_path}")
            
        return str(tmp_path)
        
    except PermissionError:
        print(f"Permission denied: Cannot access {file_path}")
        return None
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None


def check_and_install(package):
    """
        This function checks if a package is installed and installs it if not.
    """

    if importlib.util.find_spec(package) is None:
        print(f"{package} not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} has been installed")
    

__all__ = ['move_to_tmp', 'check_and_install']