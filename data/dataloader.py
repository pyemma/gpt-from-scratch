import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):

    def __init__(self, text_data: str, max_length: int, stride: int):
        self.max_length = max_length
        self.stride = stride

        self.tokenizer = tiktoken.get_encoding("gpt2")
        text_ids = self.tokenizer.encode(text_data)

        # for i in range(0, len(text_ids) - max_length, stride):
        #     input_ids = text_ids[i:i+max_length]
        #     # shift the text by 1 for the label
        #     target_ids = text_ids[i+1:i+max_length+1]
        #     self.data.append((torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)))

        # total number of samples
        num_slices = len(text_ids) // self.stride
        self.data = []
        for i in range(num_slices):
            start_idx = i * self.stride
            # shift the text by 1 for the label
            text_slice = text_ids[start_idx: min(start_idx + self.max_length + 1, len(text_ids))]
            # data, label
            self.data.append(
                (
                    torch.tensor(text_slice[:-1], dtype=torch.long), 
                    torch.tensor(text_slice[1:], dtype=torch.long)
                )
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_dataloader(text_data: str, 
                      max_length: int, 
                      stride: int, 
                      batch_size: int, 
                      drop_last: bool = True,
                      shuffle: bool = True,
                      num_workers: int = 0):
    dataset = TextDataset(text_data, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, num_workers=num_workers)