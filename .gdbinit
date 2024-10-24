# .gdbinit inside the project folder

# Enable auto-loading of Python scripts in the project directory
add-auto-load-safe-path .

# Add the current directory to the Python sys.path
python
import sys
sys.path.insert(0, ".")  # Add current directory to sys.path if not already
end


# Register type summaries for Tensor using the pretty_print function from tensor.py
python
import gdb
import re
from tools.pretty_print_lldb.parse_tensor import parse_string

class TensorPrinter:
    """Pretty-printer for rl_tools::Tensor types."""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        float_ptr = self.val["_data"]
        float_type = float_ptr.type.target()
        target = gdb.selected_inferior()

        typename = str(gdb.types.get_basic_type(self.val.type))
        print(f"Typename is: {typename}")

        tensor = parse_string(typename)
        if tensor is None:
            print(f"Parse error on: {typename}")
            return typename
        else:
            return str(tensor)

# Register the Tensor pretty-printer for specific type patterns
def register_tensor_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter("tensor_pretty_printer")
    pp.add_printer("Tensor", "^rl_tools::Tensor.*>$", TensorPrinter)
    pp.add_printer("Tensor", "^rl_tools::Tensor.*>\\s&\\s$", TensorPrinter)
    pp.add_printer("Tensor", "^const\\srl_tools::Tensor.*>$", TensorPrinter)
    pp.add_printer("Tensor", "^const\\srl_tools::Tensor.*>\\s&\\s$", TensorPrinter)
    gdb.printing.register_pretty_printer(gdb.current_objfile(), pp)

register_tensor_pretty_printer()
