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

import struct
import sys, os

render_outer_limit = os.environ["RL_TOOLS_LLDB_RENDER_OUTER_LIMIT"] if "RL_TOOLS_LLDB_RENDER_OUTER_LIMIT" in os.environ else 20
print(f"RLtools Tensor renderer outer limit: {render_outer_limit}")
render_inner_limit = os.environ["RL_TOOLS_LLDB_RENDER_INNER_LIMIT"] if "RL_TOOLS_LLDB_RENDER_INNER_LIMIT" in os.environ else 500
print(f"RLtools Tensor renderer inner limit: {render_inner_limit}")

def pad_number(number, length):
    return str(number).rjust(length)


def get_val(target, offset, float_type):
    memory = target.read_memory(offset, float_type.sizeof)
    if float_type.code == gdb.TYPE_CODE_FLT:
        if float_type.sizeof == 4:
            val = struct.unpack('f', memory)[0]
        elif float_type.sizeof == 8:
            val = struct.unpack('d', memory)[0]
        else:
            val = f"Unsupported float size: {float_type.sizeof}"
    elif float_type.code == gdb.TYPE_CODE_BOOL:
        val = struct.unpack('?', memory)[0]
    else:
        val = f"Unsupported type: {str(float_type)}"
    return val

def render_tensor(target, float_type, ptr, shape, stride, title="", use_title=False, outer_limit=render_outer_limit, inner_limit=render_inner_limit, base_offset=0):
    if len(shape) == 1:
        output = "["
        for element_i in range(shape[0]):
            pos = element_i * stride[0]
            offset = int(ptr) + base_offset + pos * float_type.sizeof
            print(f"pointer: {int(ptr)}, base_offset: {base_offset}, pos: {pos} offset: {offset} type size: {float_type.sizeof}")

            val = get_val(target, offset, float_type)
            output += str(val) + ", "
        if shape[0] > 0:
            output = output[:-2]
        return output + "]"
    elif len(shape) == 2:
        output = "[ " + (("\\\\ Subtensor: " + title + f"({shape[0]}x{shape[1]})") if use_title and len(title) > 0 else "") + "\n"
        for row_i in range(shape[0]) if shape[0] < inner_limit else list(range(inner_limit // 2)) + ["..."] + list(range(shape[0] - inner_limit // 2, shape[0])):
            if row_i == "...":
                output += "...\n"
                continue
            output += "[" if shape[0] < inner_limit else f"{row_i}: ["
            for col_i in range(shape[1]) if shape[1] < inner_limit else list(range(inner_limit // 2)) + ["..."] + list(range(shape[1] - inner_limit // 2, shape[1])):
                if col_i == "...":
                    output += "..., "
                    continue
                pos = row_i * stride[0] + col_i * stride[1]
                offset = int(ptr) + base_offset + pos * float_type.sizeof
                val = get_val(target, offset, float_type)
                output += str(val) + ", "
            if shape[1] > 0:
                output = output[:-2]
            output += "], \n"
        if shape[0] > 0:
            output = output[:-3]
            output += "\n"
        return output + "]\n"
    else:
        output = "[" + ("\n" if len(shape) == 3 else "")
        for i in range(shape[0]) if shape[0] < outer_limit else list(range(outer_limit // 2)) + ["..."] + list(range(shape[0] - outer_limit // 2, shape[0])):
            if i != "...":
                current_title = title + pad_number(i, 10) + " | "
                new_base_offset = base_offset + i * stride[0] * float_type.sizeof
                output += render_tensor(target, float_type, ptr, shape[1:], stride[1:], title=current_title, use_title=use_title, base_offset=new_base_offset) + ", \n"
            else:
                output += "...\n"
                output += "...\n"
                output += "...\n"
        if shape[0] > 0:
            output = output[:-3]
        return output + "]"


class TensorPrinter:
    """Pretty-printer for rl_tools::Tensor types."""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        float_ptr = self.val["_data"]
        float_type = float_ptr.type.target()
        target = gdb.selected_inferior()

        typename = str(gdb.types.get_basic_type(self.val.type))
        print(f"float_type is: {float_type}")
        print(f"Typename is: {typename}")

        tensor = parse_string(typename)
        if tensor is None:
            print(f"Parse error on: {typename}")
            return typename
        else:
            return str(tensor) + "\n" + render_tensor(target, float_type, float_ptr, tensor.shape, tensor.stride)


import gdb
import re
import struct
import json

def decode_row_major(typename):
    regex = r"^\s*(?:const|\s*)\s*rl_tools\s*::\s*Matrix\s*<\s*rl_tools\s*::\s*matrix\s*::\s*Specification\s*<\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*rl_tools\s*::\s*matrix\s*::\s*layouts\s*::\s*RowMajorAlignment\s*<\s*([^,]+)\s*,\s*([^,]+)\s*>\s*,\s*([^,]+)\s*>\s*>\s*(&|\s*)\s*$"
    result = re.match(regex, typename)
    if result is None:
        return None
    else:
        meta = dict(zip(["T", "TI", "ROWS", "COLS", "DYNAMIC_ALLOCATION", "TI2", "ROW_MAJOR_ALIGNMENT", "CONST"], result.groups()))
        meta["ROWS"] = int(meta["ROWS"])
        meta["COLS"] = int(meta["COLS"])
        meta["ROW_MAJOR_ALIGNMENT"] = int(meta["ROW_MAJOR_ALIGNMENT"])
        ALIGNMENT = meta["ROW_MAJOR_ALIGNMENT"]
        meta["ROW_PITCH"] = ((meta["COLS"] + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
        return meta

def decode_fixed(typename):
    regex = r"^\s*(?:const|\s*)\s*rl_tools\s*::\s*Matrix\s*<\s*rl_tools\s*::\s*matrix\s*::\s*Specification\s*<\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*rl_tools\s*::\s*matrix\s*::\s*layouts\s*::\s*Fixed\s*<\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*>\s*,\s*([^,]+)\s*>\s*>\s*(&|\s*)\s*$"
    result = re.match(regex, typename)
    if result is None:
        return None
    else:
        meta = dict(zip(["T", "TI", "ROWS", "COLS", "DYNAMIC_ALLOCATION", "TI2", "ROW_PITCH", "COL_PITCH", "CONST"], result.groups()))
        meta["ROWS"] = int(meta["ROWS"])
        meta["COLS"] = int(meta["COLS"])
        meta["ROW_PITCH"] = int(meta["ROW_PITCH"])
        meta["COL_PITCH"] = int(meta["COL_PITCH"])
        return meta

def render_matrix(target, float_type, ptr, meta, layout):
    all_data = {"meta": meta, "data": []}
    for row_i in range(meta["ROWS"]):
        current_row = []
        for col_i in range(meta["COLS"]):
            if layout == "row_major":
                pos = row_i * meta["ROW_PITCH"] + col_i
            elif layout == "fixed":
                pos = row_i * meta["ROW_PITCH"] + col_i * meta["COL_PITCH"]
            else:
                continue

            offset = int(ptr) + pos * float_type.sizeof
            val = get_val(target, offset, float_type)
            current_row.append(val)
        all_data["data"].append(current_row)
    return json.dumps(all_data)


class MatrixPrinter:
    """Pretty-printer for rl_tools::Matrix types."""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        typename = str(gdb.types.get_basic_type(self.val.type))
        _data = self.val["_data"]

        data_type_code = _data.type.code
        if data_type_code == gdb.TYPE_CODE_PTR:
            float_ptr = _data
        elif data_type_code == gdb.TYPE_CODE_ARRAY:
            float_ptr = _data[0].address
        else:
            raise ValueError(f"Unexpected _data type: {_data.type}")

        float_type = float_ptr.type.target()
        target = gdb.selected_inferior()

        # Determine the matrix layout and decode metadata
        if "RowMajorAlignment" in typename:
            meta = decode_row_major(typename)
            layout = "row_major"
        elif "Fixed" in typename:
            meta = decode_fixed(typename)
            layout = "fixed"
        else:
            return f"Unsupported matrix type: {typename}"

        # If metadata could not be parsed, return an error message
        if meta is None:
            print(f"Meta could not be parsed for: {typename}")
            return f"Matrix type could not be parsed: {typename}"

        # Render the matrix and return the result as a string
        print(f"Rendering matrix with meta: {meta} type {float_type} ptr {float_ptr}")
        return render_matrix(target, float_type, float_ptr, meta, layout)



# Register the Tensor pretty-printer for specific type patterns
def register_tensor_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter("tensor_pretty_printer")
    pp.add_printer("Tensor", "^rl_tools::Tensor.*>$", TensorPrinter)
    pp.add_printer("Tensor", "^rl_tools::Tensor.*>\\s&\\s$", TensorPrinter)
    pp.add_printer("Tensor", "^const\\srl_tools::Tensor.*>$", TensorPrinter)
    pp.add_printer("Tensor", "^const\\srl_tools::Tensor.*>\\s&\\s$", TensorPrinter)

    pp.add_printer("Matrix", "^rl_tools::Matrix<rl_tools::matrix::Specification<.*RowMajorAlignment.*>$", MatrixPrinter)
    pp.add_printer("Matrix", "^rl_tools::Matrix<rl_tools::matrix::Specification<.*Fixed.*>$", MatrixPrinter)
    pp.add_printer("Matrix", "^const\\srl_tools::Matrix<rl_tools::matrix::Specification<.*RowMajorAlignment.*>$", MatrixPrinter)
    pp.add_printer("Matrix", "^const\\srl_tools::Matrix<rl_tools::matrix::Specification<.*Fixed.*>$", MatrixPrinter)
    gdb.printing.register_pretty_printer(gdb.current_objfile(), pp)

register_tensor_pretty_printer()
