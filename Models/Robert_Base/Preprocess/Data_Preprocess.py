import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, "../", "Config")
sys.path.append(CONFIG_PATH)

from Config_Hyper import MODEL_NAME, DATA_DIR, MAX_LEN


def load_and_prepare_data():
    data_files = {
        "train": os.path.join(DATA_DIR, "viet_med-ner-train.arrow"),
        "validation": os.path.join(DATA_DIR, "viet_med-ner-validation.arrow"),
        "test": os.path.join(DATA_DIR, "viet_med-ner-test.arrow")
    }
    dataset = load_dataset("arrow", data_files=data_files)

    drop_cols = [c for c in ["audio", "duration"] if c in dataset["train"].column_names]
    if drop_cols:
        dataset = dataset.remove_columns(drop_cols)

    label_list = sorted(set(lbl for ex in dataset["train"]["labels"] for lbl in ex))
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    print("\nLabel -> id:")
    for lbl, idx in label_to_id.items():
        print(f"{idx:>2}: {lbl}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    print(f"\nLoaded tokenizer: {tokenizer.__class__.__name__}")
    print("PAD token:", tokenizer.pad_token, "| ID:", tokenizer.pad_token_id)

    # Encode label
    def encode_labels(example):
        example["label_ids"] = [label_to_id[label] for label in example["labels"]]
        return example

    dataset = dataset.map(encode_labels)

    # Tokenize + align
    def tokenize_and_align_labels(example):
        tokenized = tokenizer(
            example["words"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            is_split_into_words=True,
            return_tensors=None 
        )

        word_ids = tokenized.word_ids()
        if word_ids is None:
            word_ids = [None] * len(tokenized["input_ids"])

        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(example["label_ids"][word_idx])

        if len(label_ids) < MAX_LEN:
            label_ids += [-100] * (MAX_LEN - len(label_ids))
        else:
            label_ids = label_ids[:MAX_LEN]

        tokenized["labels"] = label_ids
        return tokenized

    tokenized_datasets = dataset.map(
        tokenize_and_align_labels,
        batched=False,
        remove_columns=dataset["train"].column_names
    )

    tokenized_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return tokenized_datasets, label_list, label_to_id, id_to_label, tokenizer


if __name__ == "__main__":
    tokenized_datasets, label_list, label_to_id, id_to_label, tokenizer = load_and_prepare_data()

    ex = tokenized_datasets["train"][5]
    print("\n--- Kiểm tra 1 mẫu ---")
    print("input_ids:", len(ex["input_ids"]), "labels:", len(ex["labels"]))
