from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, Dataset
from peft import AutoPeftModelForSequenceClassification
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import evaluate
import json
import numpy as np


# load BERT foundation model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=6,
    id2label={0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"},  
    label2id={"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5}
)

# load fine-tuned BERT model
lora_model_saved = AutoPeftModelForSequenceClassification.from_pretrained("models/lora_distil-bert", num_labels=6)


# load test data
test_data_dict = json.load(open("data/dataset_test.json"))
test_data = Dataset.from_dict(test_data_dict)

# load train data
#train_data_dict = json.load(open("data/dataset_train.json"))
#train_data = Dataset.from_dict(train_data_dict)


# tokenize test and train dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

tokenized_test_data = test_data.map(preprocess_function, batched=True)
#tokenized_test_data = tokenized_test_data.remove_columns(['text'])

#tokenized_train_data = train_data.map(preprocess_function, batched=True)
#tokenized_train_data = tokenized_train_data.remove_columns(['text'])



accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)



training_args = TrainingArguments(
            #output_dir="models/checkpoints",
            #learning_rate=2e-3,
            #per_device_train_batch_size=10,
            per_device_eval_batch_size=10,
            #num_train_epochs=3,
            #weight_decay=0.01,
            eval_strategy="epoch",
            #save_strategy="epoch",
            #load_best_model_at_end=True,
            label_names=["labels"],
            #remove_unused_columns=False,
        )


foundation_model_trainer = Trainer(
    model=model,
    args = training_args,
    #train_dataset=tokenized_train_data,
    eval_dataset=tokenized_test_data,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)
eval_results_foundation_model = foundation_model_trainer.evaluate()

print("")
print("Accuracy foundation model on test data:")
print(eval_results_foundation_model["eval_accuracy"])
print("")


lora_trainer_saved = Trainer(
    model=lora_model_saved,
    args = training_args,
    #train_dataset=tokenized_train_data,
    eval_dataset=tokenized_test_data,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)
eval_results_lora_model = lora_trainer_saved.evaluate()

print("Accuracy lora model on test data:")
print(eval_results_lora_model["eval_accuracy"])





