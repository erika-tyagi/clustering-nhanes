"""
Configuration for Jupyter notebook environment.
"""
from IPython.core import page

def set_up():
    page.page = print
