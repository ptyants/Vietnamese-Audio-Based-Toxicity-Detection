from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"A:\A _ Working\Researching\B - AIoT Lab VN\VITOSA SpeechRun"
)

CSV_PATH = BASE_DIR / "vitosa_datasets" / (
    "final_vietnamese_toxic_utterance_dataset_v5.1.2_speaker_split.csv"
)

WAV_FOLDER = Path(
    r"A:\A _ Working\Researching\B - AIoT Lab VN\VITOSA SpeechRun"
    r"\vitosa_datasets\wav_segments_v5.1"
)

OUTPUT_DIR = BASE_DIR / "EDA_v5.1.2_fig"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_AUDIO_FILES = None


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(CSV_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# 3. BASIC CHECK
# ============================================================

required_columns = [
    "audio_path",
    "toxicity"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Missing required column: {col}"
        )

print("\nToxicity distribution:")
print(df["toxicity"].value_counts(dropna=False))


# ============================================================
# 4. BUILD AUDIO PATH
# ============================================================

def resolve_audio_path(audio_path):
    """
    Resolve audio path from CSV.

    Priority:
    1. Absolute path if it exists.
    2. WAV_FOLDER / filename
    """

    if pd.isna(audio_path):
        return None

    audio_path = str(audio_path).strip()

    # Absolute path
    p = Path(audio_path)

    if p.is_absolute() and p.exists():
        return p

    # Filename inside WAV_FOLDER
    p = WAV_FOLDER / audio_path

    if p.exists():
        return p

    # In case CSV contains nested path
    filename = Path(audio_path).name
    p = WAV_FOLDER / filename

    if p.exists():
        return p

    return None


# ============================================================
# 5. RECALCULATE AUDIO DURATION
# ============================================================

print("\n" + "=" * 70)
print("RECALCULATING AUDIO DURATION")
print("=" * 70)

durations = []
missing_files = []
failed_files = []

total = len(df)

for idx, row in df.iterrows():

    if MAX_AUDIO_FILES is not None and idx >= MAX_AUDIO_FILES:
        break

    audio_path = resolve_audio_path(row["audio_path"])

    if audio_path is None:
        durations.append(np.nan)
        missing_files.append(row["audio_path"])
        continue

    try:

        # sr=None preserves original sampling rate
        audio, sr = librosa.load(
            audio_path,
            sr=None,
            mono=False
        )

        # Your dataset should be mono.
        # Still handle unexpected stereo safely.
        if audio.ndim > 1:
            audio_length = audio.shape[-1]
        else:
            audio_length = len(audio)

        duration = audio_length / sr

        durations.append(duration)

    except Exception as e:

        durations.append(np.nan)

        failed_files.append({
            "audio_path": row["audio_path"],
            "error": str(e)
        })

    if (idx + 1) % 1000 == 0:
        print(
            f"Processed {idx + 1:,}/{total:,} "
            f"({(idx + 1) / total * 100:.1f}%)"
        )


# ============================================================
# 6. HANDLE MAX_AUDIO_FILES
# ============================================================

if MAX_AUDIO_FILES is not None:

    # If only processing a subset
    df = df.iloc[:MAX_AUDIO_FILES].copy()


df["duration_recomputed"] = durations


# ============================================================
# 7. AUDIO VALIDATION
# ============================================================

valid_duration = df["duration_recomputed"].notna()

print("\n" + "=" * 70)
print("AUDIO VALIDATION")
print("=" * 70)

print(f"Total rows       : {len(df):,}")
print(f"Valid audio      : {valid_duration.sum():,}")
print(f"Missing audio    : {len(missing_files):,}")
print(f"Failed to decode : {len(failed_files):,}")


if missing_files:

    missing_df = pd.DataFrame({
        "audio_path": missing_files
    })

    missing_df.to_csv(
        OUTPUT_DIR / "missing_audio_files.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nMissing files saved to:\n"
        f"{OUTPUT_DIR / 'missing_audio_files.csv'}"
    )


if failed_files:

    failed_df = pd.DataFrame(failed_files)

    failed_df.to_csv(
        OUTPUT_DIR / "failed_audio_files.csv",
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 8. USE RECOMPUTED DURATION
# ============================================================

df_eda = df[
    df["duration_recomputed"].notna()
].copy()


# ============================================================
# 9. DURATION BINNING
# ============================================================

# Bins:
#
# <1
# 1-2
# 2-3
# ...
# 14-15
# >=15

bins = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    np.inf
]

labels = [
    "<1",
    "1-2",
    "2-3",
    "3-4",
    "4-5",
    "5-6",
    "6-7",
    "7-8",
    "8-9",
    "9-10",
    "10-11",
    "11-12",
    "12-13",
    "13-14",
    "14-15",
    ">=15"
]

df_eda["duration_bin"] = pd.cut(
    df_eda["duration_recomputed"],
    bins=bins,
    labels=labels,
    right=False,
    include_lowest=True
)


# ============================================================
# 10. CREATE DISTRIBUTION TABLE
# ============================================================

count_table = pd.crosstab(
    df_eda["duration_bin"],
    df_eda["toxicity"]
)

# Make sure both classes exist
if 0 not in count_table.columns:
    count_table[0] = 0

if 1 not in count_table.columns:
    count_table[1] = 0

count_table = count_table[[1, 0]]

count_table.columns = [
    "Toxic class",
    "Non-toxic class"
]

count_table = count_table.reindex(
    labels,
    fill_value=0
)


# ============================================================
# 11. CONVERT TO CLASS-WISE PERCENTAGE
# ============================================================

percentage_table = count_table.copy().astype(float)

for col in percentage_table.columns:

    total_class = percentage_table[col].sum()

    if total_class > 0:
        percentage_table[col] = (
            percentage_table[col] /
            total_class *
            100
        )


# ============================================================
# 12. SAVE EDA TABLE
# ============================================================

eda_table = pd.DataFrame({
    "duration_bin": labels,

    "toxic_count": count_table["Toxic class"].values,

    "toxic_percentage": percentage_table["Toxic class"].values,

    "non_toxic_count": count_table["Non-toxic class"].values,

    "non_toxic_percentage": percentage_table["Non-toxic class"].values,
})


EDA_CSV = OUTPUT_DIR / "duration_distribution_by_toxicity.csv"

eda_table.to_csv(
    EDA_CSV,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("DURATION DISTRIBUTION")
print("=" * 70)

print(
    eda_table.to_string(
        index=False,
        formatters={
            "toxic_percentage": "{:.2f}%".format,
            "non_toxic_percentage": "{:.2f}%".format
        }
    )
)

print(
    f"\nEDA table saved to:\n{EDA_CSV}"
)


# ============================================================
# 13. PLOT
# ============================================================

x = np.arange(len(labels))

width = 0.32

fig, ax = plt.subplots(
    figsize=(14, 7),
    dpi=150
)

# Toxic = Blue
bars_toxic = ax.bar(
    x - width / 2,
    percentage_table["Toxic class"],
    width,
    label="Toxic class"
)

# Non-toxic = Red
bars_non_toxic = ax.bar(
    x + width / 2,
    percentage_table["Non-toxic class"],
    width,
    label="Non-toxic class"
)


# ============================================================
# 14. FORMAT PLOT
# ============================================================

ax.set_xlabel(
    "Duration (seconds)",
    fontsize=13
)

ax.set_ylabel(
    "Data Distribution (%)",
    fontsize=13
)

ax.set_xticks(x)

ax.set_xticklabels(
    labels,
    rotation=45,
    ha="right"
)

ax.set_ylim(
    0,
    max(
        percentage_table.max().max() * 1.15,
        20
    )
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

ax.set_axisbelow(True)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 0.93),
    ncol=2,
    frameon=False,
    fontsize=12
)

plt.tight_layout()


# ============================================================
# 15. SAVE FIGURE
# ============================================================

FIG_PATH = OUTPUT_DIR / "duration_distribution_by_toxicity.png"

plt.savefig(
    FIG_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"\nFigure saved to:\n{FIG_PATH}"
)


# ============================================================
# 16. SAVE DATASET WITH RECOMPUTED DURATION
# ============================================================

UPDATED_DATASET_PATH = (
    OUTPUT_DIR /
    "dataset_with_recomputed_duration.csv"
)

df.to_csv(
    UPDATED_DATASET_PATH,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"\nUpdated dataset saved to:\n"
    f"{UPDATED_DATASET_PATH}"
)


# ============================================================
# 17. SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("DURATION SUMMARY")
print("=" * 70)

summary = (
    df_eda
    .groupby("toxicity")["duration_recomputed"]
    .agg([
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max"
    ])
)

summary.index = [
    "Non-toxic" if i == 0 else "Toxic"
    for i in summary.index
]

print(summary.round(3))


SUMMARY_PATH = OUTPUT_DIR / "duration_summary_by_toxicity.csv"

summary.to_csv(
    SUMMARY_PATH,
    encoding="utf-8-sig"
)

print(
    f"\nSummary saved to:\n{SUMMARY_PATH}"
)