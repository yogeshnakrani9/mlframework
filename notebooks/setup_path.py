# In setup_path.py in your notebooks folder
import sys
import os

def add_project_root_to_path():
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)