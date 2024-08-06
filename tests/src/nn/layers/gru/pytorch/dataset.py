import json
import zipfile
import gzip
import os
# from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm


def load_dataset(dataset_name, sequence_length):
    dataset_name = "enwik8"

    # Load the data
    print("Loading the data:")
    if dataset_name == "enwik8":
        file_name = "enwik8.small.zip"
    else:
        file_name = "00c2bfc7-57db-496e-9d5c-d62f8d8119e3.json.zip"
    if os.path.exists("/Users"):
        full_data_path = "/Users/jonas/Downloads/" + file_name
    else:
        full_data_path = "/home/jonas/Downloads/" + file_name

    if dataset_name == "enwik8":
        full_data = None
        with gzip.open(full_data_path, "r") as z:
            full_data = z.read()
    else:
        full_data = None
        with zipfile.ZipFile(full_data_path, "r") as z:
            with z.open(os.path.basename(full_data_path)[:-4]) as f:
                full_data = json.load(f)

    n_articles = 100

    if dataset_name == "enwik8":
        long_text = full_data
    else:
        print("Concatenating the dataset")
        def get_texts(data):
            for item in tqdm(data[:n_articles]):
                yield item['text']
        long_text = "\n".join(get_texts(full_data)).encode("utf-8")

    print("Chunking the dataset")
    max_length = sequence_length + 1
    chunks = [long_text[i:i + max_length] for i in tqdm(range(0, len(long_text)-max_length))]
    return chunks

    # print("Building vocab")
    # def yield_tokens(data):
    #     for chunk in data:
    #         yield chunk

    # vocab = build_vocab_from_iterator(yield_tokens(chunks), specials=['<pad>', '<unk>'])
    # vocab.set_default_index(vocab['<unk>'])

    # Data Preprocessing