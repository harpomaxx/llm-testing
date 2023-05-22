# Import necessary libraries
import datasets
import torch
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import GPT2ForSequenceClassification
from transformers import Trainer, TrainingArguments
import evaluate
import numpy as np

# Define a function to tokenize input data
def tokenize_function(examples):
    return tokenizer(examples["domain"])

# Define a function to compute metrics for evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Load the DGA detection dataset
dataset = datasets.load_dataset("harpomaxx/dga-detection")

# Set the model checkpoint
model_checkpoint = "gpt2"

# Load the tokenizer for the GPT-2 model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the datasets
tokenized_datasets = dataset.map(tokenize_function, 
                                 batched=True, 
                                 num_proc=8, 
                                 remove_columns=["domain","label"]
                                 )

# Prepare for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Rename the column "class" to "labels"
tokenized_datasets = tokenized_datasets.rename_column("class","labels")

# Load the GPT-2 model for sequence classification
model = GPT2ForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# Fix model padding token id
model.config.pad_token_id = model.config.eos_token_id

# Load the f1 metric for evaluation
metric = evaluate.load("f1")

# Set up training arguments
training_args = TrainingArguments(
    output_dir="gpt2-dga-detector",
    learning_rate=2e-5,
    optim="adamw_torch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=8,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    logging_steps = 10,
    save_total_limit = 3,
    overwrite_output_dir = True
)

# Set up the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'].shuffle(seed=42).select(range(0,8192)), # change this to the whole dataset
    eval_dataset=tokenized_datasets['test'].shuffle(seed=42).select(range(0,1024)),   # change this to the whole dataset
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
metrics = trainer.evaluate()
print(metrics)

# Save the model and tokenizer
model.save_pretrained("./CEPH/gpt2-dga-detector")
tokenizer.save_pretrained("./CEPH/gpt2-dga-detector")

