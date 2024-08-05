import json
import zipfile
import os
# from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

# Load the data
print("Loading the data:")
if os.path.exists("/Users"):
    full_data_path = "/Users/jonas/Downloads/00c2bfc7-57db-496e-9d5c-d62f8d8119e3.json.zip"
else:
    full_data_path = "/home/jonas/Downloads/00c2bfc7-57db-496e-9d5c-d62f8d8119e3.json.zip"

full_data = None
with zipfile.ZipFile(full_data_path, "r") as z:
    with z.open(os.path.basename(full_data_path)[:-4]) as f:
        full_data = json.load(f)

n_articles = 100

print("Concatenating the dataset")
def get_texts(data):
    for item in tqdm(data[:n_articles]):
        yield item['text']
long_text = "\n".join(get_texts(full_data)).encode("utf-8")

print("Chunking the dataset")
max_length = 64 + 1
chunks = [long_text[i:i + max_length] for i in tqdm(range(0, len(long_text)-max_length))]

# print("Building vocab")
# def yield_tokens(data):
#     for chunk in data:
#         yield chunk

# vocab = build_vocab_from_iterator(yield_tokens(chunks), specials=['<pad>', '<unk>'])
# vocab.set_default_index(vocab['<unk>'])

# Data Preprocessing