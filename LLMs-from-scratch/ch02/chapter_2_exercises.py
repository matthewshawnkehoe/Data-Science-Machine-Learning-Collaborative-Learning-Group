import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tiktoken
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    """
    A PyTorch Dataset for tokenizing and creating overlapping sequences from a given text using a sliding window.

    Parameters
    ----------
    txt : str
        The input text to be tokenized and processed.
    tokenizer : tiktoken.Encoding
        Tokenizer to encode the text.
    max_length : int
        The maximum length of each tokenized sequence.
    stride : int
        The step size for the sliding window to create overlapping sequences.

    Attributes
    ----------
    input_ids : list of torch.Tensor
        List of input sequences in tokenized form.
    target_ids : list of torch.Tensor
        List of target sequences, each shifted by one token from input sequences.
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the text into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # Current input sequence
            target_chunk = token_ids[i + 1: i + max_length + 1]  # Next token as target
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """
        Returns the total number of sequences.

        Returns
        -------
        int
            The number of input sequences available in the dataset.
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Retrieves the input-target pair at the specified index.

        Parameters
        ----------
        idx : int
            The index of the sequence to retrieve.

        Returns
        -------
        tuple of torch.Tensor
            A tuple containing the input sequence and its corresponding target sequence.
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, batch_size=4, max_length=256, stride=128):
    """
    Creates a PyTorch DataLoader from the input text by tokenizing it and creating a dataset.

    Parameters
    ----------
    txt : str
        The input text to be tokenized and processed.
    batch_size : int, optional
        The batch size for the DataLoader (default is 4).
    max_length : int, optional
        The maximum length of each tokenized sequence (default is 256).
    stride : int, optional
        The step size for the sliding window (default is 128).

    Returns
    -------
    DataLoader
        A PyTorch DataLoader that yields batches of tokenized sequences.
    """

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader


if __name__ == "__main__":

    # Exercise 2.1:  Byte pair encoding of unknown words

    print('\nExercise 2.1 Solution\n')

    # Initialize the GPT-2 tokenizer from tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")

    # Encode the string "Akwirw ier" into a list of integers (tokens)
    integers = tokenizer.encode("Akwirw ier")
    print('Input string: Akwirw ier')
    print(f'\nEncoded token list: {integers}\n')

    for i in integers:
        print(f"{i} -> {tokenizer.decode([i])}")

    # Decode back into 'Akwirw ier'
    print(f'\nDecoded representation of [33901, 86, 343, 86, 220, 959]: {tokenizer.decode([33901, 86, 343, 86, 220, 959])}')

    # Encode the string "Jigglypuff" into a list of integers (tokens)
    integers = tokenizer.encode("Jigglypuff")
    print('\nInput string: Jigglypuff')
    print(f'\nEncoded token list: {integers}\n')

    for i in integers:
        print(f"{i} -> {tokenizer.decode([i])}")

    # Decode back into 'Jigglypuff'
    print(f'\nDecoded representation of [41, 6950, 306, 49357]: {tokenizer.decode([41, 6950, 306, 49357])}')

    # Encode the sentence 'The lightning-fast lightning strikes lit the night brightly.' into a list of integers (tokens)
    integers = tokenizer.encode("The lightning-fast lightning strikes lit the night brightly.")
    print('\nInput string: The lightning-fast lightning strikes lit the night brightly.')
    print(f'\nEncoded token list: {integers}\n')

    for i in integers:
        print(f"{i} -> {tokenizer.decode([i])}")

    # Decode back into 'The lightning-fast lightning strikes lit the night brightly.'
    print(f'\nDecoded representation of [464, 14357, 12, 7217, 14357, 8956, 6578, 262, 1755, 35254, 13]:'
          f' {tokenizer.decode([464, 14357, 12, 7217, 14357, 8956, 6578, 262, 1755, 35254, 13])}')

    # Exercise 2.2: Data loaders with different strides and context sizes

    print('\nExercise 2.2 Solution\n')

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_text = tokenizer.encode(raw_text)

    vocab_size = 50257
    output_dim = 256
    max_len = 4
    context_length = max_len

    token_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    dataloader = create_dataloader(raw_text, batch_size=6, max_length=2, stride=2)

    for batch in dataloader:
        x, y = batch
        print("First dataloader ...")
        print("\nBatch size:", x.shape)
        print("\nInput sequences:", x)
        print("\nTarget sequences:", y)
        break

    dataloader = create_dataloader(raw_text, batch_size=4, max_length=8, stride=2)

    for batch in dataloader:
        x, y = batch
        print("\nSecond dataloader ...")
        print("\nBatch size:", x.shape)
        print("\nInput sequences:", x)
        print("\nTarget sequences:", y)
        break
