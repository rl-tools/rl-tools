import numpy as np
from datasets import load_dataset
import datasets
import h5py
from tqdm import tqdm

splits = ["train", "test"]
ds = load_dataset("mnist", keep_in_memory=True, download_mode=datasets.DownloadMode.FORCE_REDOWNLOAD)
ds = ds.with_format("numpy")

with h5py.File("mnist.hdf5", "w") as f:
    for split in splits:
        sds = ds[split]
        image_list, label_list = [], []

        for element in tqdm(sds):
            image, label = element['image'], element['label']
            image_list.append(image)
            label_list.append(label)

        images_np = np.stack(image_list)
        images_np = images_np.reshape((images_np.shape[0], -1))
        labels_np = np.array(label_list, dtype=np.uint64).reshape((-1, 1))
        g = f.create_group(split)
        g.create_dataset("inputs", data=images_np)
        g.create_dataset("labels", data=labels_np)