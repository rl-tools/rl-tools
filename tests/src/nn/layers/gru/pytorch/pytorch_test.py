import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import lightning as pl
import datetime

from dataset import dataset, collate_fn
from model import model



# Initialize WandbLogger
wandb_logger = WandbLogger(project='your_project_name')


run_date_time =  datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename=f"{run_date_time}-{{epoch:05d}}",
    save_top_k=-1,  # Save all checkpoints
    every_n_epochs=1  # Save a checkpoint every 2 epochs
)

# Create model and trainer
trainer = pl.Trainer(max_epochs=1000, accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1, logger=wandb_logger, callbacks=[checkpoint_callback])

# Create DataLoader
train_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

# Train the model
trainer.fit(model, train_loader)

