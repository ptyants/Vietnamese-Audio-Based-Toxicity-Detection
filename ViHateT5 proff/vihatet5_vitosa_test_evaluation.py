from __future__ import annotations

r"""
ViHateT5-only evaluation on ViToSA TEST split.

Input:
    A:\A _ Working\Researching\B - AIoT Lab VN\VITOSA SpeechRun\
    vitosa_datasets\final_vietnamese_toxic_utterance_dataset_v5.1.2_speaker_split.csv

Human/reference label:
    toxicity  (0 = NON_TOXIC, 1 = TOXIC)

Model label:
    ViHateT5-base-HSD applied only to transcript.

Outputs:
    ...\xucat61\
        vitosa_test_vihatet5_predictions.csv
        metrics_vihatet5.json
        confusion_matrix.csv
        classification_report.csv
        per_class_purity.csv
        metrics_summary.txt
        confusion_matrix.png

The code intentionally uses ONLY ViHateT5 for automatic labeling.
"""

from pathlib import Path
import json
import re
import math
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
INPUT_CSV = Path(r"A:\A _ Working\Researching\B - AIoT Lab VN\VITOSA SpeechRun\vitosa_datasets\final_vietnamese_toxic_utterance_dataset_v5.1.2_speaker_split.csv")

OUTPUT_DIR = INPUT_CSV.parent / "xucat61"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "tarudesu/ViHateT5-base-HSD"
PROMPT_PREFIX = "hate-speech-detection"  # Same logic as the supplied code.
TEXT_COLUMN = "transcript"
TRUE_LABEL_COLUMN = "toxicity"
SPLIT_COLUMN = "split"
TARGET_SPLIT = "test"

BATCH_SIZE = 8
MAX_INPUT_LENGTH = 256
MAX_NEW_TOKENS = 16
NUM_BEAMS = 1
DO_SAMPLE = False
RANDOM_SEED = 42

PRED_COLUMN = "vihatet5_label"
RAW_OUTPUT_COLUMN = "vihatet5_raw_output"

# ============================================================
# HELPERS
# ============================================================

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(value) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_binary_label(value) -> int | None:
    """Map the human toxicity column to 0/1 robustly."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    text = str(value).strip().lower()
    if text in {"1", "1.0", "toxic", "tox", "true", "yes", "positive"}:
        return 1
    if text in {"0", "0.0", "non-toxic", "non_toxic", "non toxic", "nontoxic", "false", "no", "negative"}:
        return 0

    try:
        x = float(text)
        if x == 0:
            return 0
        if x == 1:
            return 1
    except ValueError:
        pass

    return None


def parse_vihatet5_output(output: str) -> int:
    """
    Convert ViHateT5 text output to binary toxicity label:

        0 = NON_TOXIC
        1 = TOXIC

    ViHateT5 may generate labels such as:
        offensive
        offense
        abusive
        hateful
        hate speech
        toxic
        non-offensive
        non-toxic
        neutral
        normal

    IMPORTANT:
    Non-toxic patterns are checked FIRST so that
    'non-offensive' is not incorrectly classified as offensive.
    """

    text = normalize_text(output).lower()

    # Normalize punctuation while preserving words.
    compact = re.sub(
        r"[^a-z0-9_\- ]+",
        " ",
        text
    ).strip()

    # ============================================================
    # NON-TOXIC / NON-OFFENSIVE
    # ============================================================
    non_toxic_patterns = [
        # Toxicity terminology
        r"\bnon\s*[-_ ]?toxic\b",
        r"\bnot\s*toxic\b",
        r"\bnon\s*[-_ ]?toxicity\b",
        r"\bno\s+toxicity\b",

        # Hate speech terminology
        r"\bnon\s*[-_ ]?hate\b",
        r"\bnot\s*hate\b",
        r"\bno\s+hate\b",
        r"\bnon\s*[-_ ]?hateful\b",
        r"\bnot\s*hateful\b",

        # Offensive terminology
        r"\bnon\s*[-_ ]?offensive\b",
        r"\bnot\s*offensive\b",
        r"\bnon\s*[-_ ]?offense\b",
        r"\bnot\s*offensive\s+speech\b",

        # Other neutral forms
        r"\bneutral\b",
        r"\bnormal\b",
        r"\bclean\b",
        r"\bsafe\b",
        r"\bbenign\b",

        # Explicit binary output
        r"\b0\b",
    ]

    for pattern in non_toxic_patterns:
        if re.search(pattern, compact):
            return 0

    # ============================================================
    # TOXIC / OFFENSIVE
    # ============================================================
    toxic_patterns = [
        # Toxicity terminology
        r"\btoxic\b",
        r"\btoxicity\b",

        # Hate speech terminology
        r"\bhate\b",
        r"\bhateful\b",
        r"\bhate\s*speech\b",

        # Offensive terminology
        r"\boffensive\b",
        r"\boffense\b",
        r"\boffence\b",
        r"\boffending\b",
        r"\boffensive\s+speech\b",

        # Abuse terminology
        r"\babusive\b",
        r"\babuse\b",
        r"\binsulting\b",
        r"\binsult\b",

        # Explicit binary output
        r"\b1\b",
    ]

    for pattern in toxic_patterns:
        if re.search(pattern, compact):
            return 1

    # ============================================================
    # VIETNAMESE OUTPUTS
    # ============================================================
    toxic_vietnamese = [
        "độc hại",
        "thù ghét",
        "lời nói căm ghét",
        "xúc phạm",
        "miệt thị",
        "lăng mạ",
        "chửi",
        "công kích",
    ]

    for keyword in toxic_vietnamese:
        if keyword in text:
            return 1

    non_toxic_vietnamese = [
        "không độc hại",
        "không thù ghét",
        "không xúc phạm",
        "không miệt thị",
        "bình thường",
        "không có độc tính",
    ]

    for keyword in non_toxic_vietnamese:
        if keyword in text:
            return 0

    # ============================================================
    # FAIL LOUDLY
    # ============================================================
    raise ValueError(
        f"Could not parse ViHateT5 output: {output!r}"
    )


def load_vihatet5():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"[MODEL] {MODEL_NAME}")
    print(f"[DEVICE] {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_vihatet5(texts: list[str], tokenizer, model, device) -> tuple[list[int], list[str]]:
    preds: list[int] = []
    raws: list[str] = []
    total = len(texts)

    for start in range(0, total, BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        prompts = [f"{PROMPT_PREFIX}: {x}" for x in batch]

        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                do_sample=DO_SAMPLE,
            )

        raw_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for raw in raw_outputs:
            raw = normalize_text(raw)
            label = parse_vihatet5_output(raw)
            raws.append(raw)
            preds.append(label)

        done = min(start + len(batch), total)
        print(f"[PREDICT] {done}/{total}")

    return preds, raws


def compute_purity_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Class purity with human labels as the reference:

      Toxic purity     = TP / (TP + FP) = precision of TOXIC.
      Non-toxic purity = TN / (TN + FN) = precision of NON-TOXIC.

    This answers: among items assigned/predicted as a class, how many truly
    belong to that human-labeled class?
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    toxic_purity = tp / (tp + fp) if (tp + fp) else 0.0
    non_toxic_purity = tn / (tn + fn) if (tn + fn) else 0.0

    # Overall clustering-style purity for the two predicted groups.
    # For binary labels this is the sum of the dominant true class in each
    # predicted cluster divided by N.
    predicted_non_toxic_cluster = tn + fp
    predicted_toxic_cluster = fn + tp
    overall_purity = (
        max(tn, fp) + max(fn, tp)
    ) / len(y_true) if len(y_true) else 0.0

    return {
        "toxic_purity": float(toxic_purity),
        "non_toxic_purity": float(non_toxic_purity),
        "overall_purity": float(overall_purity),
        "predicted_non_toxic_count": int(predicted_non_toxic_cluster),
        "predicted_toxic_count": int(predicted_toxic_cluster),
    }


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    acc = accuracy_score(y_true, y_pred)
    precision_toxic = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_toxic = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_toxic = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    precision_non_toxic = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_non_toxic = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_non_toxic = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    tpr = recall_toxic

    purity = compute_purity_per_class(y_true, y_pred)

    metrics = {
        "n_samples": int(len(y_true)),
        "support_non_toxic": int(np.sum(y_true == 0)),
        "support_toxic": int(np.sum(y_true == 1)),
        "predicted_non_toxic": int(np.sum(y_pred == 0)),
        "predicted_toxic": int(np.sum(y_pred == 1)),
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_toxic": float(precision_toxic),
        "recall_toxic": float(recall_toxic),
        "f1_toxic": float(f1_toxic),
        "precision_non_toxic": float(precision_non_toxic),
        "recall_non_toxic": float(recall_non_toxic),
        "f1_non_toxic": float(f1_non_toxic),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "specificity_non_toxic": float(specificity),
        "npv_non_toxic": float(npv),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "true_positive_rate": float(tpr),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "jaccard_toxic": float(jaccard_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "purity": purity,
        "auc_roc": None,
        "auprc": None,
        "auc_note": "ViHateT5 generation does not produce a calibrated binary probability in this script, so ROC-AUC/AUPRC are not fabricated.",
    }

    cm_df = pd.DataFrame(
        cm,
        index=["TRUE_NON_TOXIC", "TRUE_TOXIC"],
        columns=["PRED_NON_TOXIC", "PRED_TOXIC"],
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["NON_TOXIC", "TOXIC"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})

    purity_df = pd.DataFrame([
        {
            "class": "NON_TOXIC",
            "human_label_value": 0,
            "purity": purity["non_toxic_purity"],
            "formula": "TN / (TN + FN)",
        },
        {
            "class": "TOXIC",
            "human_label_value": 1,
            "purity": purity["toxic_purity"],
            "formula": "TP / (TP + FP)",
        },
        {
            "class": "OVERALL",
            "human_label_value": "both",
            "purity": purity["overall_purity"],
            "formula": "sum(max true-class count in each predicted cluster)) / N",
        },
    ])

    return metrics, cm_df, report_df, purity_df


def save_confusion_matrix_plot(cm_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    values = cm_df.to_numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(values)
    ax.set_xticks([0, 1], ["Pred Non-Toxic", "Pred Toxic"])
    ax.set_yticks([0, 1], ["True Non-Toxic", "True Toxic"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Human/reference label")
    ax.set_title("ViHateT5 - ViToSA Test Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(values[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed(RANDOM_SEED)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found:\n{INPUT_CSV}\n"
            "Hãy kiểm tra đúng ổ đĩa/path Windows trước khi chạy."
        )

    print("=" * 90)
    print("ViHateT5 ONLY - ViToSA TEST EVALUATION")
    print("=" * 90)
    print(f"[INPUT]  {INPUT_CSV}")
    print(f"[OUTPUT] {OUTPUT_DIR}")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)
    print(f"[FULL DATA] rows={len(df):,}")

    required = {TEXT_COLUMN, TRUE_LABEL_COLUMN, SPLIT_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Only ViToSA test split.
    test_df = df[
        df[SPLIT_COLUMN].astype(str).str.strip().str.lower().eq(TARGET_SPLIT)
    ].copy()

    if test_df.empty:
        raise ValueError(
            f"No rows found where {SPLIT_COLUMN!r} == {TARGET_SPLIT!r}."
        )

    # Require usable transcript + toxicity reference label.
    test_df[TEXT_COLUMN] = test_df[TEXT_COLUMN].map(normalize_text)
    test_df["human_label"] = test_df[TRUE_LABEL_COLUMN].map(normalize_binary_label)
    before = len(test_df)
    test_df = test_df[
        test_df[TEXT_COLUMN].ne("") & test_df["human_label"].notna()
    ].copy()
    dropped = before - len(test_df)

    test_df["human_label"] = test_df["human_label"].astype(int)
    test_df.reset_index(drop=False, inplace=True)
    test_df.rename(columns={"index": "original_row_index"}, inplace=True)

    print(f"[TEST] rows={len(test_df):,}")
    print(f"[DROPPED] empty transcript / invalid toxicity={dropped:,}")
    print(
        "[HUMAN LABEL DISTRIBUTION]",
        test_df["human_label"].value_counts().sort_index().to_dict(),
    )

    tokenizer, model, device = load_vihatet5()

    preds, raws = predict_vihatet5(
        test_df[TEXT_COLUMN].tolist(),
        tokenizer,
        model,
        device,
    )

    test_df[PRED_COLUMN] = preds
    test_df[RAW_OUTPUT_COLUMN] = raws
    test_df["vihatet5_label_text"] = test_df[PRED_COLUMN].map(
        {0: "NON_TOXIC", 1: "TOXIC"}
    )
    test_df["label_correct"] = (
        test_df["human_label"] == test_df[PRED_COLUMN]
    )

    y_true = test_df["human_label"].to_numpy(dtype=int)
    y_pred = test_df[PRED_COLUMN].to_numpy(dtype=int)

    metrics, cm_df, report_df, purity_df = compute_all_metrics(y_true, y_pred)
    purity = metrics["purity"]

    metrics["input_csv"] = str(INPUT_CSV)
    metrics["output_dir"] = str(OUTPUT_DIR)
    metrics["model"] = MODEL_NAME
    metrics["prompt_prefix"] = PROMPT_PREFIX
    metrics["split_filter"] = f"{SPLIT_COLUMN} == {TARGET_SPLIT}"
    metrics["reference_label_column"] = TRUE_LABEL_COLUMN
    metrics["text_column"] = TEXT_COLUMN
    metrics["device"] = str(device)
    metrics["dropped_rows"] = int(dropped)

    # --------------------------------------------------------
    # SAVE ALL OUTPUTS
    # --------------------------------------------------------
    predictions_csv = OUTPUT_DIR / "vitosa_test_vihatet5_predictions.csv"
    metrics_json = OUTPUT_DIR / "metrics_vihatet5.json"
    cm_csv = OUTPUT_DIR / "confusion_matrix.csv"
    report_csv = OUTPUT_DIR / "classification_report.csv"
    purity_csv = OUTPUT_DIR / "per_class_purity.csv"
    summary_txt = OUTPUT_DIR / "metrics_summary.txt"
    cm_png = OUTPUT_DIR / "confusion_matrix.png"

    test_df.to_csv(predictions_csv, index=False, encoding="utf-8-sig")
    cm_df.to_csv(cm_csv, encoding="utf-8-sig")
    report_df.to_csv(report_csv, index=False, encoding="utf-8-sig")
    purity_df.to_csv(purity_csv, index=False, encoding="utf-8-sig")
    metrics_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_confusion_matrix_plot(cm_df, cm_png)

    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("ViHateT5 ONLY - ViToSA TEST EVALUATION\n")
        f.write("=" * 72 + "\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Prompt: {PROMPT_PREFIX}: <transcript>\n")
        f.write(f"Split: {SPLIT_COLUMN} == {TARGET_SPLIT}\n")
        f.write(f"Reference label: {TRUE_LABEL_COLUMN}\n")
        f.write(f"Samples: {metrics['n_samples']}\n")
        f.write("\nCore classification metrics\n")
        for key in [
            "accuracy",
            "balanced_accuracy",
            "precision_toxic",
            "recall_toxic",
            "f1_toxic",
            "precision_non_toxic",
            "recall_non_toxic",
            "f1_non_toxic",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
            "specificity_non_toxic",
            "npv_non_toxic",
            "false_positive_rate",
            "false_negative_rate",
            "mcc",
            "cohen_kappa",
            "jaccard_toxic",
        ]:
            f.write(f"{key}: {metrics[key]:.6f}\n")

        f.write("\nConfusion matrix\n")
        f.write("                    Pred NON_TOXIC   Pred TOXIC\n")
        f.write(f"True NON_TOXIC       {metrics['tn']:>10}   {metrics['fp']:>10}\n")
        f.write(f"True TOXIC           {metrics['fn']:>10}   {metrics['tp']:>10}\n")

        f.write("\nPurity\n")
        f.write(f"NON_TOXIC purity: {purity['non_toxic_purity']:.6f} = TN/(TN+FN)\n")
        f.write(f"TOXIC purity:     {purity['toxic_purity']:.6f} = TP/(TP+FP)\n")
        f.write(f"Overall purity:   {purity['overall_purity']:.6f}\n")

        f.write("\nAUC/PR note\n")
        f.write(metrics["auc_note"] + "\n")

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------
    print("\n" + "=" * 90)
    print("FINAL RESULTS")
    print("=" * 90)
    print(f"Samples              : {metrics['n_samples']}")
    print(f"Accuracy             : {metrics['accuracy']:.6f}")
    print(f"Balanced Accuracy    : {metrics['balanced_accuracy']:.6f}")
    print(f"Precision (TOXIC)    : {metrics['precision_toxic']:.6f}")
    print(f"Recall (TOXIC)       : {metrics['recall_toxic']:.6f}")
    print(f"F1 (TOXIC)           : {metrics['f1_toxic']:.6f}")
    print(f"Precision (NON-TOXIC): {metrics['precision_non_toxic']:.6f}")
    print(f"Recall (NON-TOXIC)   : {metrics['recall_non_toxic']:.6f}")
    print(f"F1 (NON-TOXIC)       : {metrics['f1_non_toxic']:.6f}")
    print(f"Macro F1             : {metrics['macro_f1']:.6f}")
    print(f"Weighted F1          : {metrics['weighted_f1']:.6f}")
    print(f"Specificity          : {metrics['specificity_non_toxic']:.6f}")
    print(f"MCC                  : {metrics['mcc']:.6f}")
    print(f"Cohen Kappa          : {metrics['cohen_kappa']:.6f}")
    print(f"Jaccard (TOXIC)      : {metrics['jaccard_toxic']:.6f}")
    print(f"Purity NON-TOXIC     : {purity['non_toxic_purity']:.6f}")
    print(f"Purity TOXIC         : {purity['toxic_purity']:.6f}")
    print(f"Overall Purity       : {purity['overall_purity']:.6f}")
    print("\nConfusion Matrix [rows=true, cols=pred]:")
    print(cm_df.to_string())

    print("\nSaved files:")
    for p in [predictions_csv, metrics_json, cm_csv, report_csv, purity_csv, summary_txt, cm_png]:
        print(f"  - {p}")


if __name__ == "__main__":
    main()