import os, sys
sys.path.append(os.path.dirname(__file__))

import layer_in_c
import importlib

importlib.reload(layer_in_c)

