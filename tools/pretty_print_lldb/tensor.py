import re
import lldb
import json

from parse_tensor import parse_string


def pad_number(number, length):
    return str(number).rjust(length)
def render(target, float_type, ptr, shape, stride, title="", use_title=False):
    if len(shape) == 1:
        output = "["
        for element_i in range(shape[0]):
            pos = element_i * stride[0]
            offset = ptr.GetValueAsUnsigned() + pos * float_type.GetByteSize()
            val_wrapper = target.CreateValueFromAddress("temp", lldb.SBAddress(offset, target), float_type)
            val = val_wrapper.GetValue()
            output += str(val) + ", "
        if shape[0] > 0:
            output = output[:-2]
        return output + "]"
    elif len(shape) == 2:
        output = "[ " + (("\\\\ " + title) if use_title and len(title) > 0 else "") + "\n"
        for row_i in range(shape[0]):
            output += "["
            for col_i in range(shape[1]):
                pos = row_i * stride[0] + col_i * stride[1]
                offset = ptr.GetValueAsUnsigned() + pos * float_type.GetByteSize()
                val_wrapper = target.CreateValueFromAddress("temp", lldb.SBAddress(offset, target), float_type)
                val = val_wrapper.GetValue()
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
        for i in range(shape[0]):
            current_title = title +  pad_number(i, 10) + " | "
            output += render(target, float_type, ptr, shape[1:], stride[1:], current_title) + ", \n"
        if shape[0] > 0:
            output = output[:-3]
        return output + "]"


def pretty_print(valobj, internal_dict, options):
    float_ptr = valobj.GetChildMemberWithName("_data")
    float_type = float_ptr.GetType().GetPointeeType()
    target = valobj.GetTarget()

    tensor = parse_string(valobj.type.name)
    if tensor is None:
        print(f"Parse error on: {valobj.type.name}")
        parse_string(valobj.type.name, verbose=True)

    if tensor is None:
        return valobj.type.name

    if len(tensor.shape) != len(tensor.stride):
        return str(tensor)

    print(valobj.type.name)
    if not valobj.type.name.endswith(">"):

        return str(tensor)

    return str(tensor) + "\n" + render(target, float_type, float_ptr, tensor.shape, tensor.stride)









    meta = decode_row_major(valobj)
    # if meta is None:
    #     return f"Matrix type could not be parsed: {valobj.type.name}"
    # else:
    #     acc = f"{json.dumps(meta)}\n"
    #     for row_i in range(meta["ROWS"]):
    #         for col_i in range(meta["COLS"]):
    #             pos = row_i * meta["ROW_PITCH"] + col_i
    #             offset = float_ptr.GetValueAsUnsigned() + pos * float_type.GetByteSize()
    #             val_wrapper = target.CreateValueFromAddress("temp", lldb.SBAddress(offset, target), float_type)
    #             val = val_wrapper.GetValue()
    #             acc += str(val) + ", "
    #         acc += "\n"
    #
    #
    #     return f"Matrix type: {acc}"
        # return float_ptr.Dereference()
