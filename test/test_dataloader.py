import torch
import pytest

from data.dataloader import TextDataset, create_dataloader


class TestTextDataset:

    @pytest.fixture
    def text_data(self):
        # this would be encoded to 19 tokens
        return "This is an example of text data, and we would like to test the dataloader"

    def test_text_dataset(self, text_data):
        dataset = TextDataset(text_data, max_length=4, stride=4)
        assert len(dataset) == 4
        assert torch.equal(dataset[0][0], torch.tensor([1212, 318, 281, 1672]))
        assert torch.equal(dataset[0][1], torch.tensor([318, 281, 1672, 286]))

        assert torch.equal(dataset[1][0], torch.tensor([286, 2420, 1366, 11]))
        assert torch.equal(dataset[1][1], torch.tensor([2420, 1366, 11, 290]))


class TestDataLoader:

    @pytest.fixture
    def text_data(self):
        # this would be encoded to 19 tokens
        return "This is an example of text data, and we would like to test the dataloader"

    def test_dataloader(self, text_data):
        dataloader = create_dataloader(text_data, max_length=4, stride=4, batch_size=2)

        it = iter(dataloader)
        batch = next(it)

        assert batch[0].shape == (2, 4)
        assert batch[1].shape == (2, 4)