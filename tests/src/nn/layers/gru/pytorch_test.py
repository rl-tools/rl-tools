import torch
import json
import zipfile
import os
from torch import nn
import lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

# Load the data
full_data_path = "/home/jonas/Downloads/00c2bfc7-57db-496e-9d5c-d62f8d8119e3.json.zip"
full_data = None
with zipfile.ZipFile(full_data_path, "r") as z:
    with z.open(os.path.basename(full_data_path)[:-4]) as f:
        full_data = json.load(f)

# Assume full_data is a dictionary with a key 'text' containing one long text
long_text = full_data[0]['text']

# Split the long text into smaller chunks (e.g., sentences or fixed-size chunks)
max_length = 100
chunks = [long_text[i:i + max_length] for i in range(0, len(long_text)-max_length)]

# Build vocabulary
def yield_tokens(data):
    for chunk in data:
        yield chunk.split()

vocab = build_vocab_from_iterator(yield_tokens(chunks), specials=['<pad>', '<unk>'])
vocab.set_default_index(vocab['<unk>'])

# Data Preprocessing
class TextDataset(Dataset):
    def __init__(self, data, vocab, max_length=100):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = [self.vocab[token] for token in text.split()][:self.max_length]
        return torch.tensor(tokens), len(tokens)

def collate_fn(batch):
    texts, lengths = zip(*batch)
    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    return texts, torch.tensor(lengths)

# Create dataset
dataset = TextDataset(chunks, vocab)

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
        text, text_lengths = text.to(self.device), text_lengths.to(self.device)
        output = self(text, text_lengths)
        loss = nn.functional.cross_entropy(output, torch.zeros(output.shape[0], dtype=torch.long, device=self.device))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

# Create model and trainer
model = GRURNN(len(vocab), embedding_dim=100, hidden_dim=256, output_dim=2)
trainer = pl.Trainer(max_epochs=10, accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1)

# Create DataLoader
train_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# Train the model
trainer.fit(model, train_loader)

