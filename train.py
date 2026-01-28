import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch.nn as nn
from transformers import AutoModel

#load and flatten data
df = pd.read_csv("dataset/wikipedia.csv")

texts = []
labels = []

for _, row in df.iterrows():
    texts.append(row["human_text"])
    labels.append(0)
    texts.append(row["ai_text"])
    labels.append(1)

labels = torch.tensor(labels)


#setup yokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

enc = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=256,  #keeping smaller for my CPU
    return_tensors="pt"
)

input_ids = enc["input_ids"]
attention_mask = enc["attention_mask"]

#split into train and val

torch.manual_seed(42)

N = len(labels)
perm = torch.randperm(N)

split = int(0.8 * N)

train_idx = perm[:split]
val_idx = perm[split:]

X_train_ids = input_ids[train_idx]
X_train_mask = attention_mask[train_idx]
y_train = labels[train_idx]

X_val_ids = input_ids[val_idx]
X_val_mask = attention_mask[val_idx]
y_val = labels[val_idx]


#setup model
class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        self.fc = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.fc(cls)


#train

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = BertClassifier().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

BATCH_SIZE = 4  #keping small for my cpu

#train functions
def train_epoch():
    model.train()
    total_loss = 0

    for i in range(0, len(y_train), BATCH_SIZE):
        ids = X_train_ids[i:i+BATCH_SIZE].to(device)
        mask = X_train_mask[i:i+BATCH_SIZE].to(device)
        labels = y_train[i:i+BATCH_SIZE].to(device)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / (len(y_train) // BATCH_SIZE)


#validation loop

def eval_epoch():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(y_val), BATCH_SIZE):
            ids = X_val_ids[i:i+BATCH_SIZE].to(device)
            mask = X_val_mask[i:i+BATCH_SIZE].to(device)
            labels = y_val[i:i+BATCH_SIZE].to(device)

            logits = model(ids, mask)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

#training
for epoch in range(3):
    loss = train_epoch()
    acc = eval_epoch()
    print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")

#prediction

def predict(text):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(ids, mask)

    return "AI" if torch.argmax(logits, dim=1).item() == 1 else "Human"
