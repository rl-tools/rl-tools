import h5py


f = h5py.File('tests/nn/test.hdf5', 'r')

f.keys()
f["three_layer_fc"].keys()
f["three_layer_fc"]["age"][:]
f["three_layer_fc"]["layer_1"].keys()
f["three_layer_fc"]["layer_1"]["weights"][:].shape
f["three_layer_fc"]["layer_1"]["biases"][:].shape
f["three_layer_fc"]["layer_1"]["d_weights"][:].shape
f["three_layer_fc"]["layer_1"]["d_biases"][:].shape
f["three_layer_fc"]["layer_1"]["d_weights_first_order_moment"][:].shape
f["three_layer_fc"]["layer_1"]["d_biases_first_order_moment"][:].shape
f["three_layer_fc"]["layer_1"]["d_weights_second_order_moment"][:].shape
f["three_layer_fc"]["layer_1"]["d_biases_second_order_moment"][:].shape
f["three_layer_fc"]["layer_1"]["output"][:].shape
