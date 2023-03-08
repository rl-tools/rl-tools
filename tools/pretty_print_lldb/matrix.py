import re
import lldb


# workflow
# lldb cmake-build-debug/tests/test_rl_algorithms_td3_second_stage_mlp
# breakpoint set -f include/layer_in_c/containers/operations_generic.h -l 125
# run
# type summary clear
# command script import tools/pretty_print_lldb/matrix.py
# type summary add -F matrix.pp_matrix -x "^layer_in_c::Matrix<layer_in_c::matrix::Specification<"
# p m1

def pp_matrix(valobj, internal_dict, options):
    # regex = r"layer_in_c::Matrix<layer_in_c::matrix::Specification<([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*, layer_in_c::matrix::layouts::RowMajorAlignment<([^,]+)\s*,\s*([^(,)]+)>\s*,\s*([^,]+)>\s>"
    # regex = r"^(?:const|)\s*layer_in_c::Matrix<layer_in_c::matrix::Specification<\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*layer_in_c::matrix::layouts::RowMajorAlignment\s*<\s*([^,]+)\s*,\s*([^(,)]+)\s*>\s*,\s*([^,]+)\s*>\s>$"
    # regex = r"^(?:const|\s*)\s*layer_in_c::Matrix<layer_in_c::matrix::Specification<\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*layer_in_c::matrix::layouts::RowMajorAlignment\s*<\s*([^,]+)\s*,\s*([^,]+)\s*>\s*,\s*([^,]+)\s*>\s*>$"

    regex = r"^\s*(?:const|\s*)\s*layer_in_c\s*::\s*Matrix\s*<\s*layer_in_c\s*::\s*matrix\s*::\s*Specification\s*<\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*layer_in_c\s*::\s*matrix\s*::\s*layouts\s*::\s*RowMajorAlignment\s*<\s*([^,]+)\s*,\s*([^,]+)\s*>\s*,\s*([^,]+)\s*>\s*>\s*$"
    result = re.match(regex, valobj.type.name)
    if result is None:
        return f"Matrix type could not be parsed: {valobj.type.name}"
    else:
        meta = dict(zip(["T", "TI", "ROWS", "COLS", "TI2", "ROW_MAJOR_ALIGNMENT", "IS_VIEW"], result.groups()))
        meta["ROWS"] = int(meta["ROWS"])
        meta["COLS"] = int(meta["COLS"])
        meta["ROW_MAJOR_ALIGNMENT"] = int(meta["ROW_MAJOR_ALIGNMENT"])
        ALIGNMENT = meta["ROW_MAJOR_ALIGNMENT"]
        meta["ROW_PITCH"] = ((meta["COLS"] + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT;
        float_ptr = valobj.GetChildMemberWithName("_data")
        float_type = float_ptr.GetType().GetPointeeType()
        target = valobj.GetTarget()

        acc = "\n"
        for row_i in range(meta["ROWS"]):
            for col_i in range(meta["COLS"]):
                pos = row_i * meta["ROW_PITCH"] + col_i
                offset = float_ptr.GetValueAsUnsigned() + pos * float_type.GetByteSize()
                val_wrapper = target.CreateValueFromAddress("temp", lldb.SBAddress(offset, target), float_type)
                val = val_wrapper.GetValue()
                acc += str(val) + ", "
            acc += "\n"


        return f"Matrix type: {acc}"

# class SyntheticChildrenProvider:
#     def __init__(self, valobj, internal_dict):
#         regex = r"^\s*(?:const|\s*)\s*layer_in_c\s*::\s*Matrix\s*<\s*layer_in_c\s*::\s*matrix\s*::\s*Specification\s*<\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*layer_in_c\s*::\s*matrix\s*::\s*layouts\s*::\s*RowMajorAlignment\s*<\s*([^,]+)\s*,\s*([^,]+)\s*>\s*,\s*([^,]+)\s*>\s*>\s*$"
#         self.result = re.match(regex, valobj.type.name)
#         self.meta = dict(zip(["T", "TI", "ROWS", "COLS", "TI2", "ROW_MAJOR_ALIGNMENT", "IS_VIEW"], self.result.groups())) if self.result is not None else None
#     def num_children(self):
#         return self.meta["ROWS"] if self.meta is not None else 0
#     def get_child_index(self,name):
#         return -1;
#     def get_child_at_index(self,index):
#         pos = index
#     def update(self):
#         this call should be used to update the internal state of this Python object whenever the state of the variables in LLDB changes.[1]
#         Also, this method is invoked before any other method in the interface.
#     def has_children(self):
#         this call should return True if this object might have children, and False if this object can be guaranteed not to have children.[2]
#     def get_value(self):
#         this call can return an SBValue to be presented as the value of the synthetic value under consideration.[3]