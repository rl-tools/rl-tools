from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch
import functools
from dataset import load_dataset
from tqdm import tqdm
class TextDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.raw_data = data
        self.max_length = sequence_length + 1
        print("Preprocessing the data")
        self.data = torch.zeros((len(data), self.max_length), dtype=torch.int64)
        for i in tqdm(range(len(data))):
            text = self.raw_data[i]
            self.data[i, :] = torch.tensor(list(text), dtype=torch.int64)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :-1], self.data[idx, 1:]

def collate_fn(batch):
    inputs, targets = zip(*batch)
    input_lengths = [len(seq) for seq in inputs]
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded, torch.tensor(input_lengths)

def create_dataset(dataset_name, sequence_length):
    chunks = load_dataset(dataset_name, sequence_length)
    return TextDataset(chunks, sequence_length)