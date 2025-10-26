import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import glob
import re

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, "../Config")
sys.path.append(CONFIG_PATH)

from Config_Hyper import OUTPUT_DIR


def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def main():
    sample_text = (
        "Mẹ tôi 50 tuổi bị béo phì đến viện ngày 15 tháng 09 với triệu chứng ho và đau đầu. "
        "Sử dụng thuốc Ibuprofen và thiết bị đo huyết áp tại bệnh viện Đa khoa Hà Nội."
    )

    checkpoint_dirs = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
    if not checkpoint_dirs:
        print("Không tìm thấy checkpoint nào trong OUTPUT_DIR !!!")
        return
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
    best_ckpt = checkpoint_dirs[-1]
    print(f"Load model tốt nhất từ: {best_ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(best_ckpt)
    model = AutoModelForTokenClassification.from_pretrained(best_ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    model.to(device)
    model.eval()
    
    nlp_ner = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )

    sentences = split_sentences(sample_text)
    label_colors = {}
    base_colors = ["\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m"]

    print("\nKết quả NER theo từng câu :\n")

    for sent_idx, sentence in enumerate(sentences, 1):
        print(f"\n──────────────────────────────")
        print(f"Câu {sent_idx}: {sentence}")
        
        encoding = tokenizer(sentence, return_tensors="pt", truncation=True)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

        tokens = tokens[1:-1]
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].detach().cpu().tolist()[1:-1]

        print(f"  - > Tokenized ({len(tokens)} tokens): {tokens}")

        id2label = model.config.id2label

        print("\n  - > Token | Label | Label_ID")
        print("  --------------------------------------------")
        for tok, label_id in zip(tokens, predictions):
            label_name = id2label.get(label_id, "O")
            print(f"  {tok:15s} | {label_name:20s} | {label_id}")

        ner_results = nlp_ner(sentence)

        for i, res in enumerate(ner_results):
            label_colors[res['entity_group']] = base_colors[i % len(base_colors)]

        print("\n  - > Kết quả NER gộp:")
        if ner_results:
            for res in ner_results:
                entity = res['word']
                label = res['entity_group']
                color = label_colors.get(label, "\033[0m")
                print(f"    - > {color}\033[1m{entity}\033[0m ({label})")
        else:
            print("    - > Không phát hiện thực thể nào.")
        print()


if __name__ == "__main__":
    main()
