import logging

import tiktoken
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader


class SpamDataset(Dataset):

    def __init__(self, df, max_length = None, pad_token_id = 50256):
        super().__init__()

        self.df = df
        self.df["label"] = self.df["label"].apply(lambda x: 1 if x == "spam" else 0)

        # encoding the text
        tokenizer = tiktoken.get_encoding("gpt2")
        self.encoded_text = [
            tokenizer.encode(text) for text in df["text"]
        ]

        # padding the text to the longest text
        self.max_length = max_length or max([len(text) for text in self.encoded_text])
        self.encoded_text = [
            torch.nn.functional.pad(torch.tensor(text, dtype=torch.long), (0, self.max_length - len(text)), "constant", pad_token_id)
            for text in self.encoded_text
        ]

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        tokens = self.encoded_text[idx]
        label = self.df.iloc[idx]["label"]
        return tokens, torch.tensor(label, dtype=torch.long)

def create_dataloader(file_path: str, batch_size: int, num_workers: int = 0):
    # load the dataset and split into train/val/test
    df = pd.read_csv(file_path, sep="\t", header=None, names=["label", "text"])

    # handle unbalanced dataset
    num_spam = df[df["label"] == "spam"].shape[0]
    ham_subset = df[df["label"] == "ham"].sample(num_spam)
    df = pd.concat([ham_subset, df[df["label"] == "spam"]])

    train_df, test_df = train_test_split(df, test_size=0.2)  # 0.8 train, 0.2 test
    train_df, val_df = train_test_split(train_df, test_size=0.25)  # 0.6 train, 0.2 val
    
    logging.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # reset the index to avoid old index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_dataset = SpamDataset(train_df)
    val_dataset = SpamDataset(val_df)
    test_dataset = SpamDataset(test_df)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    )


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloader("./sms-spam-collection/SMSSpamCollection", batch_size=2, num_workers=0)

    it = iter(train_loader)
    print(next(it))