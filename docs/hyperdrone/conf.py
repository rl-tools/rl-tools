import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))
project = "HyperDrone"

extensions = ["myst_nb", "IPython.sphinxext.ipython_console_highlighting", "shiki_highlight"]
nb_execution_mode = "force"
nb_execution_in_temp = True
nb_execution_timeout = -1
nb_execution_raise_on_error = True
include_patterns = ["index.rst", "src/HyperLuxMinimal.ipynb"]

html_theme = "furo"
html_static_path = ["_static"]
