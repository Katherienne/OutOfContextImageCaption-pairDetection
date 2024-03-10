# !pip install -U sentence-transformers
# !pip install evaluate
# ! pip install wandb

##Requirement
import torch
import pandas as pd
import random
import numpy as np
import functools
import pickle
import torch.nn as nn
import json
import evaluate
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer,util
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim import AdamW
from transformers import get_scheduler


##Train model
###Load train dataset
class OutContextData(Dataset):
    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path
        self.__load_file()

    def __load_file(self):
        df = pd.read_csv(self.data_path)
        self.__raw_data = df.to_dict('records')
        self.__raw_len = len(self.__raw_data)

    def __len__(self) -> int:
        return self.__raw_len

    def __getitem__(self, idx):
        d_point = self.__raw_data[idx]
        concatenated_caption = d_point['combined_caption']      
        label = int(d_point['label'])
        
        return {
            "text": concatenated_caption,
            "label": label,
        }

###Load evaluate dataset
class LoadTest(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            self.data = [json.loads(line) for line in file]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        def extract_mid(caption):
            if len(caption) > 128:
                start = max(0, len(caption) // 2 - 64)
                end = min(len(caption), len(caption) // 2 + 64)
                return caption[start:end]
            return caption

        caption1 = extract_mid(item.get('caption1', ''))
        caption2 = extract_mid(item.get('caption2', ''))
        concatenated_caption = f"{caption1} {caption2}"

        label = int(item.get('context_label', 0))

        return {
            'img_local_path': item.get('img_local_path'),
            'text': concatenated_caption,
            'label': label,
        }

### Custome Train dataset
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        return {
            'label': torch.tensor(label),
            'text': text,
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze()
        }


    def forward(self, model, input_ids, attention_mask, labels):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            pooled_output = outputs.pooler_output
        pooled_output.requires_grad = True
        return pooled_output


def collate_fn(batch):
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}





###Load model
from transformers import AutoModelForSequenceClassification

model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

## get data and dataloader from path

train_path = "/kaggle/input/data-unique-img/new_train.csv"
train_dataset = OutContextData(train_path)

eval_path = "/kaggle/input/test-new/test_data.json"
eval_dataset = LoadTest(eval_path)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

dataset = CustomDataset(train_dataset, tokenizer)
train_dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

eval_data = CustomDataset(eval_dataset, tokenizer)
eval_dataloader = DataLoader(eval_data, batch_size=32, collate_fn=collate_fn)

## defined optimized

optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

num_epochs = 20
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

## Training

progress_bar = tqdm(range(num_training_steps))
best_acc = 0.8
best_model_path = "best_model.pt"


for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        inputs = {key: val for key, val in batch.items() if key != 'labels'}
        labels = batch['labels']
        
        
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()
        #         logits = dataset.forward(model, input_ids, attention_mask, labels)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)

    predictions_list = []
    labels_list = []
    eval_loss = 0.0
    
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        logits = outputs.logits

        predictions = torch.argmax(logits, dim=-1)
        predictions_list.extend(predictions.cpu().tolist())
        labels_list.extend(batch["labels"].cpu().tolist())
        eval_loss += criterion(outputs.logits, batch['labels']).item()

    accuracy = accuracy_score(labels_list, predictions_list)
    precision = precision_score(labels_list, predictions_list, average='weighted')
    recall = recall_score(labels_list, predictions_list, average='weighted')
    f1 = f1_score(labels_list, predictions_list, average='weighted')

    print("Evaluation Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print(f"Eval Loss: {eval_loss / len(eval_dataloader):.4f}")
    print("===============")
    
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), best_model_path)
        print("Best model saved with acc score:", best_acc) 


# Load best model 
output_model = 'BEST MODEL PATH'
checkpoint = torch.load(output_model, map_location='cpu')

model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
saved_state_dict = torch.load('BEST MODEL PATH', map_location=torch.device('cpu'))
torch.save(model.state_dict(), 'best_model.pt')

num_classes_saved = saved_state_dict['classifier.out_proj.weight'].shape[0]
num_classes_current = model.config.num_labels

if num_classes_saved != num_classes_current:
    model.config.num_labels = num_classes_saved
    model.classifier.out_proj = torch.nn.Linear(model.config.hidden_size, num_classes_saved)

model.load_state_dict(saved_state_dict, strict=False)

