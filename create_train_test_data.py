# Fine-tuning dataset:
# https://huggingface.co/datasets/dair-ai/emotion


from datasets import load_dataset
import json

# train/test split
splits = ["train", "test"]
ds = {split: ds for split, ds in zip(splits, load_dataset("emotion", split=splits))}
dataset_train = ds["train"].shuffle(seed=42).select(range(2000))
dataset_test = ds["test"].select(range(500))

# save data
with open("data/dataset_train.json", 'w') as f:
    json.dump(dataset_train.to_dict(), f) 
print("train dataset saved")
    
with open("data/dataset_test.json", 'w') as f:
    json.dump(dataset_test.to_dict(), f)
print("test dataset saved")
