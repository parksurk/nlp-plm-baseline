import torch
from torch.utils.data import Dataset


class TextClassificationCollator():

    def __init__(self, tokenizer, max_length, with_text=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text

    def __call__(self, samples):
        texts = [s['text'] for s in samples]
        labels = [s['label'] for s in samples]

        # Debugging prints
        # print(f"Processing batch with {len(texts)} samples.")
        # for i, text in enumerate(texts):
        #     print(f"Sample {i}: {text[:50]}...")  # Print first 50 chars of the text for checking


        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        )

        return_value = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long),
        }
        if self.with_text:
            return_value['text'] = texts

        return return_value


class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        # Additional check for unexpected types
        if not isinstance(text, str) or not isinstance(label, int):
            raise ValueError(f"Invalid data at index {item}: text={text}, label={label}")

        #print(f'Returning: {{ "text": {text}, "label": {label} }}')

        return {
            'text': text,
            'label': label,
        }
