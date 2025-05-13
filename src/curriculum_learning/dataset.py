import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Sample configuration
TOKENIZER_NAME = "bert-base-uncased"
MAX_SEQ_LENGTH = 128

# Mock dataset class
class TextbookDataset(Dataset):
    def __init__(self, grade, data_dir="data", tokenizer=None):
        self.grade = grade
        self.data_dir = data_dir
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        
        # Load sample texts (replace this with actual data loading)
        grade_dir = os.path.join(data_dir, f"grade_{grade}")
        text_file = os.path.join(grade_dir, "text_samples.txt")
        
        # Mock data - replace with actual file reading
        self.examples = [
            "The cat sat on the mat.",  # Simple sentence for lower grades
            "2 + 2 = 4",                # Math problem
            "Plants need sunlight to grow.",  # Science fact
            # ... more examples
        ]
        
        # Real implementation would read from text_file:
        # with open(text_file, "r") as f:
        #     self.examples = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # Tokenize text (using MLM-style formatting)
        encoding = self.tokenizer(
            text,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # For language modeling, create masked labels
        input_ids = encoding.input_ids.squeeze()
        labels = input_ids.clone()
        
        # Mask 15% of tokens (adjust for your task)
        mask_probability = torch.full(labels.shape, 0.15)
        masked_indices = torch.bernoulli(mask_probability).bool()
        labels[~masked_indices] = -100  # Ignore loss for non-masked tokens
        
        return {
            "input_ids": input_ids,
            "attention_mask": encoding.attention_mask.squeeze(),
            "labels": labels
        }

# Collate function for DataLoader
def collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch])
    }

# Function to get dataloader for a specific grade
def get_dataloader_for_grade(grade, batch_size=32):
    dataset = TextbookDataset(grade=grade)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )