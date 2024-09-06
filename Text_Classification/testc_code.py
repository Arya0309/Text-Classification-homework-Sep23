import torch
import transformers
import datasets
import os
import json
import numpy as np

# from pathlib import Path
from datasets import load_dataset

train = datasets.load_dataset("ag_news", split="train")

random_seed = 42  # To insure reproducibility（重現性）
splits = train.train_test_split(test_size=0.2, seed=random_seed)
train, valid = splits["train"], splits["test"]

test = datasets.load_dataset("ag_news", split="test")

tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")


def to_torch_data(hug_dataset):
    dataset = hug_dataset.map(
        lambda batch: tokenizer(
            batch["text"], 
            truncation=True, 
            padding='max_length',
            max_length=512
        ),
        batched=True,
    )
    dataset.set_format(
        type="torch", 
        columns=[
            "input_ids", 
            "attention_mask", 
            "label"
        ]
    )
    return dataset


train_dataset = to_torch_data(train)
val_dataset = to_torch_data(valid)
test_dataset = to_torch_data(test)

model = transformers.AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=4
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="micro"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


training_args = transformers.TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=2,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="tensorboard",
    save_total_limit=10,
    logging_dir="./logs",
    logging_steps=10,
    seed=random_seed,
)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.args._n_gpu = 2

trainer.train()
trainer.predict(test_dataset)
