import tiktoken
import torch

class DataLoader:
    
    def __init__(self, batch_size: int, seq_length: int, file_name: str):
        self.batch_size: int = batch_size
        self.seq_length: int = seq_length
        self.file_name: str = file_name
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.current_position: int = 0

        with open(file_name, 'r') as f:
            data = f.read()
        # Encode the data using tiktoken tokenizer
        encoded_data = self.tokenizer.encode(data)
        self.tokens: int = torch.tensor(encoded_data)
        print(f"Number of tokens in dataset: {len(encoded_data)}")
        print(f"Number of batches in dataset: {len(encoded_data) // (self.batch_size * self.seq_length)}")

    def next_batch(self):
        first_id: int = self.current_position
        last_id: int = self.current_position + self.batch_size * self.seq_length + 1
        buffer: torch.Tensor = self.tokens[first_id : last_id]
        inputs: torch.Tensor = buffer[:-1].view(self.batch_size, self.seq_length)
        targets: torch.Tensor = buffer[1:].view(self.batch_size, self.seq_length)

        self.current_position += self.batch_size * self.seq_length

        # if current_position + self.batch_size * self.seq_length > len(self.tokens):
            
        return inputs, targets

            
