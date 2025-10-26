import os
import sys
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import evaluate
import json
import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import (
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback
)
from transformers.trainer_utils import IntervalStrategy, EvalPrediction

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, "../", "Config")
PREPROCESS_PATH = os.path.join(CURRENT_DIR, "../", "Preprocess")
sys.path.extend([CONFIG_PATH, PREPROCESS_PATH])

from Config_Hyper_Vihealth import (
    MODEL_NAME, BATCH_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY,
    WARMUP_RATIO, GRAD_ACCUM_STEPS, NUM_WORKERS, FP16, MAX_GRAD_NORM,
    SAVE_TOTAL_LIMIT, LOGGING_STEPS, METRIC_FOR_BEST_MODEL, OUTPUT_DIR,
    LABEL_SMOOTHING, LR_SCHEDULER_TYPE, OPTIMIZER,
    EARLY_STOPPING_PATIENCE, SEED
)

from Data_Preprocess import load_and_prepare_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    set_seed(42)
    logger.info(f"Bắt đầu train với model: {MODEL_NAME}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)}GB")
            torch.backends.cudnn.benchmark = True
        except Exception:
            logger.info("Không lấy được thông tin GPU chi tiết.")
    else:
        logger.warning("Chạy trên CPU (ko có GPU) !!!")

    tokenized_datasets, label_list, label2id, id2label, tokenizer = load_and_prepare_data()
    logger.info(f"Train={len(tokenized_datasets['train'])}, Val={len(tokenized_datasets['validation'])}, labels={len(label_list)}")

    # Nếu tokenizer không có pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Tokenizer không có pad_token, gán pad_token = eos_token ({tokenizer.pad_token})")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info("Tokenizer không có pad_token và eos_token, đã thêm [PAD] làm pad_token")


    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label={int(k): str(v) for k, v in id2label.items()},
        label2id={str(k): int(v) for k, v in label2id.items()} if not isinstance(list(label2id.keys())[0], int) else label2id,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.1
    )

    metric = evaluate.load("seqeval")

    def compute_metrics(eval_pred: EvalPrediction):
        """
        eval_pred: tuple (predictions, label_ids)
        predictions có thể là logits hoặc tuple => lấy phần logits
        map id -> label name, bỏ -100
        """
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.argmax(preds, axis=-1)

        true_preds, true_labels = [], []
        for p_row, l_row in zip(preds, labels):
            p_tags, g_tags = [], []
            for p_id, l_id in zip(p_row, l_row):
                if l_id == -100:
                    continue
                p_id = int(p_id)
                l_id = int(l_id)
                # map id -> label name
                p_tags.append(id2label.get(p_id, str(p_id)))
                g_tags.append(id2label.get(l_id, str(l_id)))
            true_preds.append(p_tags)
            true_labels.append(g_tags)

        results = metric.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": results.get("overall_precision"),
            "recall": results.get("overall_recall"),
            "f1": results.get("overall_f1"),
            "accuracy": results.get("overall_accuracy"),
        }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=LOGGING_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=True,
        fp16=FP16,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        dataloader_num_workers=NUM_WORKERS,
        group_by_length=True,
        max_grad_norm=MAX_GRAD_NORM,
        report_to="none",
        label_smoothing_factor=LABEL_SMOOTHING,  
        seed=SEED,                                      
        lr_scheduler_type=LR_SCHEDULER_TYPE,           
        warmup_steps=0,
        optim=OPTIMIZER, 
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    )


    logger.info("Bắt đầu huấn luyện...")
    trainer.train()
    logger.info("Huấn luyện hoàn tất.")

    checkpoint_dirs = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
    if checkpoint_dirs:
        checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
        best_ckpt = checkpoint_dirs[-1]
        trainer_state_file = os.path.join(best_ckpt, "trainer_state.json")
        if os.path.exists(trainer_state_file):
            with open(trainer_state_file, "r") as f:
                trainer_state = json.load(f)
            logs = trainer_state.get("log_history", [])
            logger.info(f"Loaded log_history từ best checkpoint: {best_ckpt}")
        else:
            logger.warning("trainer_state.json không tồn tại trong checkpoint, dùng log_history hiện tại")
            logs = trainer.state.log_history
    else:
        logger.warning("Không tìm thấy checkpoint, dùng log_history trong Trainer")
        logs = trainer.state.log_history


    epoch_losses = {}
    epochs = []
    val_loss = []
    f1 = []
    precision = []
    recall = []

    for record in logs:
        epoch_float = record.get("epoch", 0)
        if "loss" in record:
            epoch_int = int(epoch_float) + 1
            epoch_losses.setdefault(epoch_int, []).append(record["loss"])
        if "eval_loss" in record:
            epoch_int = int(epoch_float)
            if epoch_int not in epochs:
                epochs.append(epoch_int)
                val_loss.append(record.get("eval_loss", np.nan))
                f1.append(record.get("eval_f1", np.nan))
                precision.append(record.get("eval_precision", np.nan))
                recall.append(record.get("eval_recall", np.nan))

    train_loss_epoch = [np.mean(epoch_losses.get(ep, [np.nan])) for ep in epochs]

    plt.figure(figsize=(12,5))
    plt.plot(epochs, train_loss_epoch, label='train_loss', marker='o')
    plt.plot(epochs, val_loss, label='val_loss', marker='o')
    plt.plot(epochs, f1, label='f1', marker='o')
    plt.plot(epochs, precision, label='precision', marker='o')
    plt.plot(epochs, recall, label='recall', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Score / Loss")
    plt.title("Training Metrics per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "training_metrics.png"))
    plt.close()
    logger.info("Training metrics chart lưu thành công!")

    if checkpoint_dirs:
        best_model = AutoModelForTokenClassification.from_pretrained(best_ckpt).to(device)
        trainer.model = best_model
    else:
        logger.warning("Không có checkpoint, dùng model hiện tại trong trainer.")

    logger.info("Chạy dự đoán trên tập validation để tạo confusion matrix...")
    pred_output = trainer.predict(tokenized_datasets["validation"])
    preds = np.argmax(pred_output.predictions, axis=-1)
    labels = pred_output.label_ids

    all_preds, all_labels = [], []
    for p_row, l_row in zip(preds, labels):
        for p_id, l_id in zip(p_row, l_row):
            if l_id == -100:
                continue
            all_preds.append(int(p_id))
            all_labels.append(int(l_id))

    num_labels = len(label_list)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_labels)))

    fig, ax = plt.subplots(figsize=(max(8, num_labels*0.6), max(6, num_labels*0.6)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[id2label[i] for i in range(num_labels)])
    disp.plot(include_values=True, cmap="Blues", ax=ax, xticks_rotation=45, values_format='d')

    plt.title("Confusion Matrix từ Best Checkpoint", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_val.png"), dpi=200)
    plt.close()
    logger.info("Confusion matrix lưu thành công !!!")


if __name__ == "__main__":
    main()
