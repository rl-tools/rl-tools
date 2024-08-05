import numpy as np

from dataset import max_length, chunks


data = np.zeros((len(chunks), max_length), dtype=np.int64)
for i in range(len(data)):
    text = chunks[i]
    data[i, :] = np.array(list(text), dtype=np.int64)
dataset = data[:, :-1], data[:, 1:]

def iterate_batches(batch_size):
    inputs, targets = dataset
    s = 0
    while True:
        if s == 0:
            # Reset permutation:
            perm = np.random.permutation(inputs.shape[0])
        ids = perm[s : s + batch_size]
        yield inputs[ids], targets[ids]
        s += batch_size
        if s >= inputs.shape[0]:
            s = 0