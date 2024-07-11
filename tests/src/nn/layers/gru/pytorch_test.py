import torch
import json
import zipfile
import os



full_data_path = "/Users/jonas/Downloads/00c2bfc7-57db-496e-9d5c-d62f8d8119e3.json.zip"

full_data = None
with zipfile.ZipFile(full_data_path, "r") as z:
    with z.open(os.path.basename(full_data_path)[:-4]) as f:
        full_data = json.load(f)


import torch
from torch import nn
import lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

# Data Preprocessing
class TextDataset(Dataset):
    def __init__(self, data, vocab, max_length=100):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        tokens = [self.vocab[token] for token in text.split()][:self.max_length]
        return torch.tensor(tokens), len(tokens)

def collate_fn(batch):
    texts, lengths = zip(*batch)
    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    return texts, torch.tensor(lengths)

data = full_data[0]

# Build vocabulary
def yield_tokens(data):
    yield data['text'].split()

vocab = build_vocab_from_iterator(yield_tokens(data), specials=['<pad>', '<unk>'])
vocab.set_default_index(vocab['<unk>'])

# Create dataset
dataset = TextDataset([data], vocab)

# PyTorch Lightning Module
class GRURNN(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed_embedded)
        return self.fc(hidden.squeeze(0))

    def training_step(self, batch, batch_idx):
        text, text_lengths = batch
        output = self(text, text_lengths)
        loss = nn.functional.cross_entropy(output, torch.zeros(output.shape[0], dtype=torch.long))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

# Create model and trainer
model = GRURNN(len(vocab), embedding_dim=100, hidden_dim=256, output_dim=2)
trainer = pl.Trainer(max_epochs=10)

# Create DataLoader
train_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# Train the model
trainer.fit(model, train_loader)