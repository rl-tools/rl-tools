from torch import nn
import torch
import lightning as pl
from dataset import vocab

pad_index = vocab.get_stoi()['<pad>']

# PyTorch Lightning Module
class GRURNN(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        gru_output, hidden = self.gru(embedded)
        output = self.fc(gru_output)
        return output

    def training_step(self, batch, batch_idx):
        text, targets, text_lengths = batch
        text, text_lengths = text.to(self.device), text_lengths.to(self.device)
        output = self(text, text_lengths)
        loss = nn.functional.cross_entropy(output.view(-1, output.shape[-1]), targets.view(-1), ignore_index=pad_index)
        perplexity = torch.exp(loss)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_perplexity', perplexity, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


model = GRURNN(len(vocab), embedding_dim=32, hidden_dim=256, output_dim=len(vocab), num_layers=4)