from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
import json
from peft import LoraConfig, get_peft_model



# load foundation BERT model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=6,
    id2label={0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"},  
    label2id={"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5}
)

# load train and test data
train_data_dict = json.load(open("data/dataset_train.json"))
train_data = Dataset.from_dict(train_data_dict)

test_data_dict = json.load(open("data/dataset_test.json"))
test_data = Dataset.from_dict(test_data_dict)


# tokenize dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

tokenized_train_data = train_data.map(preprocess_function, batched=True)
tokenized_train_data = tokenized_train_data.remove_columns(['text'])
tokenized_test_data = test_data.map(preprocess_function, batched=True)
tokenized_test_data = tokenized_test_data.remove_columns(['text'])


# define metrics
def compute_accuracy(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


# define lora config
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_lin", "k_lin", "v_lin"],  # layers to train
    task_type='SEQ_CLS',
    lora_dropout=0.1,
    bias="none",
    modules_to_save=[],
)

# define lora model and trainer
lora_model = get_peft_model(model, lora_config)

lora_trainer = Trainer(
    model=lora_model,
    args=TrainingArguments(
        output_dir="models/emotion_classification",
        learning_rate=2e-3,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        label_names=["labels"],
        remove_unused_columns=False,
    ),
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_test_data,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_accuracy,
)

# train lora model
lora_trainer.train()

# save fine-tuned lora model
lora_model.save_pretrained("models/lora_distil-bert")