import os
import sys
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from transformers import AutoModelForTokenClassification, Trainer
import evaluate
import glob

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, "../", "Config")
PREPROCESS_PATH = os.path.join(CURRENT_DIR, "../", "Preprocess")
sys.path.extend([CONFIG_PATH, PREPROCESS_PATH])

from Config_Hyper import OUTPUT_DIR
from Data_Preprocess import load_and_prepare_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Sử dụng device: {device}")

    tokenized_datasets, label_list, label2id, id2label, tokenizer = load_and_prepare_data()
    test_dataset = tokenized_datasets.get("test")
    if test_dataset is None:
        logger.error("Không tìm thấy test dataset!")
        return

    num_labels = len(label_list)

    checkpoint_dirs = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
    if not checkpoint_dirs:
        logger.error("Không tìm thấy checkpoint nào trong OUTPUT_DIR!")
        return
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
    best_ckpt = checkpoint_dirs[-1]
    logger.info(f"Load model tốt nhất từ: {best_ckpt}")

    model = AutoModelForTokenClassification.from_pretrained(
        best_ckpt,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    ).to(device)

    trainer = Trainer(model=model, tokenizer=tokenizer)

    logger.info("Bắt đầu predict trên test set...")
    pred_output = trainer.predict(test_dataset)
    preds = np.argmax(pred_output.predictions, axis=-1)
    labels = pred_output.label_ids

    all_preds, all_labels = [], []
    for p_row, l_row in zip(preds, labels):
        for p_id, l_id in zip(p_row, l_row):
            if l_id == -100:
                continue
            all_preds.append(int(p_id))
            all_labels.append(int(l_id))

    true_preds = [id2label[p] for p in all_preds]
    true_labels = [id2label[l] for l in all_labels]

    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=[true_preds], references=[true_labels])
    logger.info(f"Test set metrics (seqeval): {results}")

    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(num_labels)), zero_division=0
    )
    logger.info("Chi tiết per-class metrics:")
    for i, label in enumerate(label_list):
        logger.info(f"{label:15}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")

    # Tính macro,micro,weighted
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(num_labels)), average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(num_labels)), average='micro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(num_labels)), average='weighted', zero_division=0
    )

    logger.info(f"\n=== Average Metrics ===")
    logger.info(f"Macro     : Precision={precision_macro:.4f}, Recall={recall_macro:.4f}, F1={f1_macro:.4f}")
    logger.info(f"Micro     : Precision={precision_micro:.4f}, Recall={recall_micro:.4f}, F1={f1_micro:.4f}")
    logger.info(f"Weighted  : Precision={precision_weighted:.4f}, Recall={recall_weighted:.4f}, F1={f1_weighted:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_labels)))
    fig, ax = plt.subplots(figsize=(max(10, num_labels*0.8), max(8, num_labels*0.8)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)
    disp.plot(include_values=True, cmap="Blues", ax=ax, xticks_rotation=45, values_format='d')

    plt.title("Confusion Matrix trên Test Set", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(fontsize=12, rotation=45, ha="right")
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_test.png"), dpi=200)
    plt.close()
    logger.info("Confusion matrix test set lưu thành công!")

if __name__ == "__main__":
    main()
