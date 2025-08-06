# Parameter-efficient Fine-Tuning of Transformer Encoder Foundation Model

- base model: distilbert-bert-uncased (+ classifier layer)
- peft method: LoRA
- downstream task: emotion classification
- fine-tuning dataset: https://huggingface.co/datasets/dair-ai/emotion

<br>

## Create environment

- Install the virtual environment and the required packages (`macOS`):

    ```
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
<br>

## Create train/test datasets

```
python create_train_test_data.py
```
<br>

## Fine-tune the foundation model

```
python train.py
```
<br>

## Evaluation
- eval metrics: accuracy
- comparison of foundation model and fine-tuned model

```
python eval.py
```
<br>

## Evaluation results
- foundation model: acc 0.144
- fine-tuned model: acc 0.884

<br>

## ToDo
- for better comparison, train classifyer layer of foundation model