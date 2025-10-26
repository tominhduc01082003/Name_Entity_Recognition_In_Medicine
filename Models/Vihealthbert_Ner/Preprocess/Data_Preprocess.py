import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, "../", "Config")
sys.path.append(CONFIG_PATH)

from Config_Hyper_Vihealth import MODEL_NAME, DATA_DIR, MAX_LEN


def load_and_prepare_data():
    """
    - Nạp dữ liệu từ file .arrow (train/val/test)
    - Mã hóa nhãn thành chỉ số
    - Tokenize + align labels (do ViHealthBERT không có word_ids())
    - Trả về dataset ở dạng torch Tensor sẵn sàng cho huấn luyện
    """

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

    print("\n=== Danh sách nhãn (Label -> ID) ===")
    for lbl, idx in label_to_id.items():
        print(f"{idx:>2}: {lbl}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    print(f"\nLoaded tokenizer: {tokenizer.__class__.__name__}")
    print("PAD token:", tokenizer.pad_token, "| ID:", tokenizer.pad_token_id)

    def encode_labels(example):
        example["label_ids"] = [label_to_id[label] for label in example["labels"]]
        return example

    dataset = dataset.map(encode_labels)

    def tokenize_and_align_labels(example):
        """
        Mỗi từ được tokenize riêng, tất cả sub-token của từ đó cùng nhãn.
        Nếu sub-token rỗng, dùng <unk> để thay thế.
        """

        words = example["words"]
        word_label_ids = example["label_ids"]

        all_subtokens = []
        all_labels = []

        for word, label_id in zip(words, word_label_ids):
            sub_toks = tokenizer.tokenize(word)

            if len(sub_toks) == 0:
                sub_toks = [tokenizer.unk_token or word]

            all_subtokens.extend(sub_toks)
            all_labels.extend([label_id] * len(sub_toks))

        #sub-token - > id
        ids_no_special = tokenizer.convert_tokens_to_ids(all_subtokens)

        #Giới hạn độ dài < MAX_LEN
        num_special = tokenizer.num_special_tokens_to_add(pair=False)
        max_subtok = MAX_LEN - num_special
        if len(ids_no_special) > max_subtok:
            ids_no_special = ids_no_special[:max_subtok]
            all_labels = all_labels[:max_subtok]

        # Thêm special tokens ([CLS], [SEP], …)
        input_ids = tokenizer.build_inputs_with_special_tokens(ids_no_special)
        n_special = len(input_ids) - len(ids_no_special)
        n_spec_before = 1 if n_special >= 1 else 0
        n_spec_after = n_special - n_spec_before

        # Gán nhãn -100 cho token đặc biệt
        labels_with_special = ([-100] * n_spec_before) + all_labels + ([-100] * n_spec_after)
        attention_mask = [1] * len(input_ids)

        # Pad cho đủ MAX_LEN
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        pad_length = MAX_LEN - len(input_ids)
        if pad_length > 0:
            input_ids += [pad_id] * pad_length
            attention_mask += [0] * pad_length
            labels_with_special += [-100] * pad_length
        else:
            input_ids = input_ids[:MAX_LEN]
            attention_mask = attention_mask[:MAX_LEN]
            labels_with_special = labels_with_special[:MAX_LEN]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_with_special
        }

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
    print("In kiểm tra mẫu đầu tiên (tập train)")
    ex = tokenized_datasets["train"][0]
    tokens = tokenizer.convert_ids_to_tokens(ex["input_ids"])
    label_ids = [int(x) for x in ex["labels"]]
    attn_mask = [int(x) for x in ex["attention_mask"]]
    print(f"Số lượng token: {len(tokens)}")
    print(f"Chiều dài MAX_LEN: {len(ex['input_ids'])}")

    print("\n=== TOKEN    |     LABEL_ID |  LABEL_NAME  | MASK ===")
    for tok, lbl_id, mask in zip(tokens, label_ids, attn_mask):
        if lbl_id == -100:
            label_name = "IGNORED"
        else:
            label_name = id_to_label.get(lbl_id, str(lbl_id))
        print(f"{tok:15s} | {lbl_id:>8d} | {label_name:15s} | mask={mask}")

