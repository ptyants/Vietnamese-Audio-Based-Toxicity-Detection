# ============================================================
# COMPLETE EDA - VIETNAMESE AUDIO TOXIC UTTERANCE DATASET
# ============================================================

from pathlib import Path
import ast
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIG
# ============================================================

BASE_DIR = Path(
    r"A:\A _ Working\Researching\B - AIoT Lab VN\VITOSA SpeechRun"
)

CSV_PATH = BASE_DIR / (
    "final_vietnamese_toxic_utterance_dataset_v5.1.2_speaker_split.csv"
)

WAV_FOLDER = Path(
    r"A:\A _ Working\Researching\B - AIoT Lab VN\VITOSA SpeechRun"
    r"\vitosa_datasets\wav_segments_v5.1"
)

OUTPUT_DIR = BASE_DIR / "EDA_v5.1.2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_AUDIO_FILES = None
N_AUDIO_VISUAL_SAMPLES = 3
AUTO_FIND_CSV = True


# ============================================================
# 2. AUDIO LIBRARIES
# ============================================================

try:
    import librosa
    import librosa.display
    import soundfile as sf
except ImportError:
    raise ImportError(
        "Thiếu thư viện audio. Chạy:\n"
        "pip install librosa soundfile"
    )


# ============================================================
# 3. HELPERS
# ============================================================

def print_section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def save_csv(dataframe, filename):
    path = OUTPUT_DIR / filename
    dataframe.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )
    print(f"[SAVED CSV] {path}")


def save_figure(filename):
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )
    print(f"[SAVED PNG] {path}")
    plt.show()
    plt.close()


def parse_list(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:
        result = ast.literal_eval(str(value))
        return result if isinstance(result, list) else []
    except Exception:
        return []


# ============================================================
# 4. FIND CSV
# ============================================================

print_section("1. LOCATING DATASET")

if not CSV_PATH.exists() and AUTO_FIND_CSV:
    candidates = list(
        BASE_DIR.rglob(
            "final_vietnamese_toxic_utterance_dataset_v5.1.2_speaker_split.csv"
        )
    )

    if not candidates:
        candidates = list(
            BASE_DIR.rglob("*v5.1.2*speaker_split*.csv")
        )

    if candidates:
        CSV_PATH = candidates[0]

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy CSV:\n{CSV_PATH}"
    )

print(f"[OK] CSV: {CSV_PATH}")


# ============================================================
# 5. LOAD CSV
# ============================================================

print_section("2. LOAD DATASET")

df = pd.read_csv(CSV_PATH)

print("trước lọc")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")
print(f"Memory  : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Lọc bỏ external_non_toxic
df = df[df["split"] != "external_non_toxic"].copy()

# Reset index
df.reset_index(drop=True, inplace=True)
print("sau lọc:")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")
print(f"Memory  : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nColumns:")
for col in df.columns:
    print(f"  - {col}")


# ============================================================
# 6. BASIC QUALITY
# ============================================================

print_section("3. BASIC DATA QUALITY")

dtype_report = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "missing": [int(df[c].isna().sum()) for c in df.columns],
    "missing_pct": [float(df[c].isna().mean()) for c in df.columns],
    "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
})

print(dtype_report.to_string(index=False))
save_csv(dtype_report, "01_dtype_missing_report.csv")


# ============================================================
# 7. DUPLICATES
# ============================================================

print_section("4. DUPLICATE ANALYSIS")

print(
    f"Full-row duplicates: "
    f"{df.duplicated().sum():,}"
)

if "audio_path" in df.columns:
    duplicate_audio = df[
        df["audio_path"].duplicated(keep=False)
    ].sort_values("audio_path")

    print(
        f"audio_path duplicate rows: "
        f"{len(duplicate_audio):,}"
    )

    save_csv(
        duplicate_audio,
        "02_duplicate_audio_paths.csv"
    )


# ============================================================
# 8. SOURCE
# ============================================================

if "audio_source" in df.columns:

    print_section("5. AUDIO SOURCE DISTRIBUTION")

    source_stats = (
        df["audio_source"]
        .value_counts(dropna=False)
        .rename_axis("audio_source")
        .reset_index(name="count")
    )

    source_stats["percentage"] = (
        source_stats["count"] / len(df)
    )

    print(source_stats.to_string(index=False))

    save_csv(
        source_stats,
        "03_audio_source_distribution.csv"
    )

    plt.figure(figsize=(10, 5))
    plt.bar(
        source_stats["audio_source"].astype(str),
        source_stats["count"]
    )
    plt.xlabel("Audio source")
    plt.ylabel("Number of utterances")
    plt.title("Dataset Distribution by Audio Source")
    plt.xticks(rotation=45, ha="right")
    save_figure("03_audio_source_distribution.png")


# ============================================================
# 9. TOXICITY
# ============================================================

if "toxicity" in df.columns:

    print_section("6. TOXICITY DISTRIBUTION")

    tox_stats = (
        df["toxicity"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("toxicity")
        .reset_index(name="count")
    )

    tox_stats["percentage"] = (
        tox_stats["count"] / len(df)
    )

    print(tox_stats.to_string(index=False))
    save_csv(tox_stats, "04_toxicity_distribution.csv")

    plt.figure(figsize=(7, 5))
    plt.bar(
        ["Non-toxic (0)", "Toxic (1)"],
        [
            int((df["toxicity"] == 0).sum()),
            int((df["toxicity"] == 1).sum()),
        ]
    )
    plt.ylabel("Number of utterances")
    plt.title("Overall Toxicity Distribution")
    save_figure("04_toxicity_distribution.png")


# ============================================================
# 10. SPLIT
# ============================================================

if "split" in df.columns:

    print_section("7. SPLIT DISTRIBUTION")

    split_stats = (
        df["split"]
        .value_counts(dropna=False)
        .rename_axis("split")
        .reset_index(name="count")
    )

    split_stats["percentage"] = (
        split_stats["count"] / len(df)
    )

    print(split_stats.to_string(index=False))
    save_csv(split_stats, "05_split_distribution.csv")

    plt.figure(figsize=(9, 5))
    plt.bar(
        split_stats["split"].astype(str),
        split_stats["count"]
    )
    plt.xlabel("Split")
    plt.ylabel("Number of utterances")
    plt.title("Dataset Split Distribution")
    plt.xticks(rotation=25, ha="right")
    save_figure("05_split_distribution.png")


# ============================================================
# 11. SPLIT x TOXICITY
# ============================================================

if {"split", "toxicity"}.issubset(df.columns):

    print_section("8. SPLIT x TOXICITY")

    count_table = pd.crosstab(
        df["split"],
        df["toxicity"]
    )

    pct_table = pd.crosstab(
        df["split"],
        df["toxicity"],
        normalize="index"
    )

    print("Count:")
    print(count_table)

    print("\nPercentage:")
    print(pct_table.round(4))

    save_csv(
        count_table.reset_index(),
        "06_split_x_toxicity_count.csv"
    )

    save_csv(
        pct_table.reset_index(),
        "06_split_x_toxicity_percentage.csv"
    )

    count_table.plot(
        kind="bar",
        stacked=True,
        figsize=(9, 5)
    )

    plt.xlabel("Split")
    plt.ylabel("Number of utterances")
    plt.title("Toxic / Non-toxic Distribution by Split")
    plt.xticks(rotation=20, ha="right")
    save_figure("06_split_x_toxicity.png")


# ============================================================
# 12. SOURCE x SPLIT
# ============================================================

if {"audio_source", "split"}.issubset(df.columns):

    print_section("9. AUDIO SOURCE x SPLIT")

    table = pd.crosstab(
        df["audio_source"],
        df["split"]
    )

    print(table)
    save_csv(table.reset_index(), "07_source_x_split.csv")

    table.plot(
        kind="bar",
        stacked=True,
        figsize=(11, 6)
    )

    plt.xlabel("Audio source")
    plt.ylabel("Number of utterances")
    plt.title("Audio Source Distribution across Splits")
    plt.xticks(rotation=45, ha="right")
    save_figure("07_source_x_split.png")


# ============================================================
# 13. SOURCE x TOXICITY
# ============================================================

if {"audio_source", "toxicity"}.issubset(df.columns):

    print_section("10. AUDIO SOURCE x TOXICITY")

    table = pd.crosstab(
        df["audio_source"],
        df["toxicity"]
    )

    pct_table = pd.crosstab(
        df["audio_source"],
        df["toxicity"],
        normalize="index"
    )

    print("Count:")
    print(table)

    print("\nPercentage:")
    print(pct_table.round(4))

    save_csv(
        table.reset_index(),
        "08_source_x_toxicity_count.csv"
    )

    save_csv(
        pct_table.reset_index(),
        "08_source_x_toxicity_percentage.csv"
    )

    table.plot(
        kind="bar",
        stacked=True,
        figsize=(11, 6)
    )

    plt.xlabel("Audio source")
    plt.ylabel("Number of utterances")
    plt.title("Toxic / Non-toxic Distribution by Audio Source")
    plt.xticks(rotation=45, ha="right")
    save_figure("08_source_x_toxicity.png")


# ============================================================
# 14. CSV DURATION
# ============================================================

if "duration" in df.columns:

    print_section("11. CSV DURATION")

    duration = pd.to_numeric(
        df["duration"],
        errors="coerce"
    )

    print(duration.describe())

    save_csv(
        duration.describe().to_frame().reset_index(),
        "10_csv_duration_statistics.csv"
    )

    plt.figure(figsize=(9, 5))
    plt.hist(
        duration.dropna(),
        bins=60
    )
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of utterances")
    plt.title("CSV Duration Distribution")
    save_figure("10_csv_duration_distribution.png")


# ============================================================
# 15. TRANSCRIPT
# ============================================================

if "transcript" in df.columns:

    print_section("12. TRANSCRIPT ANALYSIS")

    transcript = (
        df["transcript"]
        .fillna("")
        .astype(str)
    )

    char_len = transcript.str.len()
    word_len = transcript.str.split().str.len()

    print(
        f"Missing transcript: "
        f"{df['transcript'].isna().sum():,}"
    )

    print(
        f"Empty transcript: "
        f"{(transcript.str.strip() == '').sum():,}"
    )

    print("\nCharacter length:")
    print(char_len.describe())

    print("\nWord length:")
    print(word_len.describe())

    transcript_stats = pd.DataFrame({
        "transcript_char_len": char_len,
        "transcript_word_len": word_len,
    })

    if "toxicity" in df.columns:
        transcript_stats["toxicity"] = df["toxicity"]

    if "audio_source" in df.columns:
        transcript_stats["audio_source"] = df["audio_source"]

    save_csv(
        transcript_stats,
        "11_transcript_statistics.csv"
    )

    plt.figure(figsize=(9, 5))
    plt.hist(word_len, bins=60)
    plt.xlabel("Number of words")
    plt.ylabel("Number of utterances")
    plt.title("Transcript Length Distribution")
    save_figure("11_transcript_word_distribution.png")


# ============================================================
# 16. TOXIC SPANS / BIO
# ============================================================

if {"toxicity", "toxic_spans_text"}.issubset(df.columns):

    print_section("13. TOXIC SPAN CONSISTENCY")

    spans = df["toxic_spans_text"].apply(parse_list)
    has_span = spans.apply(lambda x: len(x) > 0)

    table = pd.crosstab(
        df["toxicity"],
        has_span
    )

    print(table)
    save_csv(
        table.reset_index(),
        "12_toxicity_x_toxic_span.csv"
    )


if {"toxicity", "BIO_Tag"}.issubset(df.columns):

    tags = df["BIO_Tag"].apply(parse_list)
    has_bio = tags.apply(lambda x: len(x) > 0)

    table = pd.crosstab(
        df["toxicity"],
        has_bio
    )

    save_csv(
        table.reset_index(),
        "12_toxicity_x_bio.csv"
    )


# ============================================================
# 17. SPEAKER INDEPENDENCE
# ============================================================

SPEAKER_COL = "pseudo_speaker_id"

if {SPEAKER_COL, "split"}.issubset(df.columns):

    print_section("14. SPEAKER-INDEPENDENT ANALYSIS")

    print(
        f"Missing speaker: "
        f"{df[SPEAKER_COL].isna().sum():,}"
    )

    print(
        f"Unique speakers: "
        f"{df[SPEAKER_COL].nunique(dropna=True):,}"
    )

    speaker_sets = {}

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        speaker_sets[split_name] = set(
            df.loc[
                (df["split"] == split_name)
                & df[SPEAKER_COL].notna(),
                SPEAKER_COL
            ]
        )

    rows = []

    split_names = [
        "train",
        "validation",
        "test"
    ]

    for i, a in enumerate(split_names):

        for b in split_names[i + 1:]:

            overlap = (
                speaker_sets[a]
                & speaker_sets[b]
            )

            rows.append({
                "split_a": a,
                "split_b": b,
                "overlap_speakers": len(overlap),
                "examples": ";".join(
                    list(overlap)[:20]
                )
            })

    speaker_overlap = pd.DataFrame(rows)

    print(speaker_overlap)

    save_csv(
        speaker_overlap,
        "13_speaker_overlap.csv"
    )


# ============================================================
# 18. FIND WAV
# ============================================================

print_section("15. REAL WAV AUDIO ANALYSIS")

if not WAV_FOLDER.exists():
    raise FileNotFoundError(
        f"Không tìm thấy WAV_FOLDER:\n{WAV_FOLDER}"
    )

wav_files = sorted(
    WAV_FOLDER.rglob("*.wav")
)

print(f"WAV folder: {WAV_FOLDER}")
print(f"Found WAV : {len(wav_files):,}")

if not wav_files:
    raise RuntimeError(
        "Không tìm thấy file .wav trong WAV_FOLDER."
    )

if MAX_AUDIO_FILES is not None:
    wav_files = wav_files[:MAX_AUDIO_FILES]
    print(
        f"Processing limit: "
        f"{MAX_AUDIO_FILES:,}"
    )


# ============================================================
# 19. BUILD AUDIO_DF
# ============================================================

audio_rows = []
corrupted_rows = []

for i, wav_path in enumerate(
    wav_files,
    start=1
):

    try:

        info = sf.info(
            str(wav_path)
        )

        y, sr = librosa.load(
            str(wav_path),
            sr=None,
            mono=True
        )

        if len(y) == 0:
            raise ValueError("Empty audio")

        duration_sec = len(y) / sr

        rms = float(
            np.sqrt(
                np.mean(
                    np.square(y)
                )
            )
        )

        peak_amplitude = float(
            np.max(
                np.abs(y)
            )
        )

        zero_crossing_rate = float(
            np.mean(
                librosa.feature.zero_crossing_rate(
                    y,
                    frame_length=2048,
                    hop_length=512
                )
            )
        )

        spectral_centroid_hz = float(
            np.mean(
                librosa.feature.spectral_centroid(
                    y=y,
                    sr=sr
                )
            )
        )

        rms_frames = (
            librosa.feature.rms(y=y)[0]
        )

        if len(rms_frames):

            silence_threshold = max(
                1e-5,
                float(
                    np.max(rms_frames)
                ) * 0.02
            )

            silence_ratio = float(
                np.mean(
                    rms_frames
                    <= silence_threshold
                )
            )

        else:

            silence_ratio = np.nan

        audio_rows.append({

            "file_name":
                wav_path.name,

            "relative_path":
                str(
                    wav_path.relative_to(
                        WAV_FOLDER
                    )
                ),

            "sample_rate":
                int(info.samplerate),

            "channels":
                int(info.channels),

            "frames":
                int(info.frames),

            "duration_sec":
                duration_sec,

            "rms":
                rms,

            "peak_amplitude":
                peak_amplitude,

            "zero_crossing_rate":
                zero_crossing_rate,

            "spectral_centroid_hz":
                spectral_centroid_hz,

            "silence_ratio":
                silence_ratio,

            "subtype":
                info.subtype,
        })

    except Exception as e:

        corrupted_rows.append({
            "file_name": wav_path.name,
            "path": str(wav_path),
            "error": repr(e),
        })

    if (
        i % 500 == 0
        or i == len(wav_files)
    ):

        print(
            f"Processed "
            f"{i:,}/{len(wav_files):,}"
        )


# ============================================================
# 20. AUDIO DATAFRAME
# ============================================================

audio_df = pd.DataFrame(
    audio_rows
)

corrupted_df = pd.DataFrame(
    corrupted_rows
)

print_section(
    "16. AUDIO DATAFRAME"
)

print(
    f"audio_df shape : {audio_df.shape}"
)

print(
    f"Readable WAV   : {len(audio_df):,}"
)

print(
    f"Corrupted WAV  : {len(corrupted_df):,}"
)

if audio_df.empty:
    raise RuntimeError(
        "audio_df vẫn rỗng. Không đọc được WAV."
    )

print(
    "\nColumns:"
)

print(
    audio_df.columns.tolist()
)

print(
    "\nFirst 5 rows:"
)

print(
    audio_df.head()
)

save_csv(
    audio_df,
    "16_audio_signal_statistics.csv"
)

save_csv(
    corrupted_df,
    "16_corrupted_audio_files.csv"
)


# ============================================================
# 21. AUDIO STATISTICS
# ============================================================

print_section(
    "17. AUDIO SIGNAL STATISTICS"
)

audio_features = [
    "duration_sec",
    "rms",
    "peak_amplitude",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "silence_ratio",
]

audio_statistics = (
    audio_df[audio_features]
    .describe()
    .T
    .reset_index()
    .rename(columns={"index": "feature"})
)

print(
    audio_statistics.to_string(
        index=False
    )
)

save_csv(
    audio_statistics,
    "17_audio_statistics.csv"
)


# ============================================================
# 22. HISTOGRAM
# ============================================================

print_section(
    "18. AUDIO HISTOGRAMS"
)

for feature in audio_features:

    values = pd.to_numeric(
        audio_df[feature],
        errors="coerce"
    )

    values = (
        values
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if values.empty:
        continue

    plt.figure(
        figsize=(9, 5)
    )

    plt.hist(
        values,
        bins=60
    )

    plt.xlabel(feature)
    plt.ylabel("Number of WAV files")
    plt.title(
        f"Distribution - {feature}"
    )

    save_figure(
        f"20_histogram_{feature}.png"
    )


# ============================================================
# 23. BOX PLOT + IQR OUTLIER
# ============================================================

print_section(
    "19. BOX PLOTS - AUDIO FEATURES"
)

boxplot_results = []

for feature in audio_features:

    print(
        f"\nProcessing: {feature}"
    )

    values = pd.to_numeric(
        audio_df[feature],
        errors="coerce"
    )

    values = (
        values
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    print(
        f"Valid values: {len(values):,}"
    )

    if values.empty:
        print("[SKIP] No valid data.")
        continue

    q1 = float(
        values.quantile(0.25)
    )

    median = float(
        values.median()
    )

    q3 = float(
        values.quantile(0.75)
    )

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (values < lower_bound)
        |
        (values > upper_bound)
    )

    outlier_count = int(
        outlier_mask.sum()
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.boxplot(
        values,
        vert=True,
        showfliers=True
    )

    plt.title(
        f"Box Plot - {feature}"
    )

    plt.ylabel(feature)

    plt.xticks(
        [1],
        [feature]
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    save_figure(
        f"30_boxplot_{feature}.png"
    )

    boxplot_results.append({

        "feature":
            feature,

        "count":
            len(values),

        "q1":
            q1,

        "median":
            median,

        "q3":
            q3,

        "iqr":
            iqr,

        "lower_bound":
            lower_bound,

        "upper_bound":
            upper_bound,

        "outlier_count":
            outlier_count,

        "outlier_percentage":
            outlier_count / len(values),
    })


boxplot_report = pd.DataFrame(
    boxplot_results
)

save_csv(
    boxplot_report,
    "30_boxplot_iqr_outlier_report.csv"
)

print(
    "\nIQR Outlier Report:"
)

print(
    boxplot_report.to_string(
        index=False
    )
)


# ============================================================
# 24. SAMPLE RATE
# ============================================================

print_section(
    "20. AUDIO SAMPLING RATE"
)

sr_stats = (
    audio_df["sample_rate"]
    .value_counts()
    .sort_index()
    .rename_axis("sample_rate")
    .reset_index(name="count")
)

print(sr_stats)
save_csv(
    sr_stats,
    "31_audio_sampling_rate.csv"
)

plt.figure(figsize=(8, 5))

plt.bar(
    sr_stats["sample_rate"].astype(str),
    sr_stats["count"]
)

plt.xlabel("Sampling rate (Hz)")
plt.ylabel("Number of WAV files")
plt.title("Actual WAV Sampling Rate")

save_figure(
    "31_audio_sampling_rate.png"
)


# ============================================================
# 25. CHANNELS
# ============================================================

channel_stats = (
    audio_df["channels"]
    .value_counts()
    .sort_index()
    .rename_axis("channels")
    .reset_index(name="count")
)

save_csv(
    channel_stats,
    "32_audio_channels.csv"
)

plt.figure(figsize=(7, 5))

plt.bar(
    channel_stats["channels"].astype(str),
    channel_stats["count"]
)

plt.xlabel("Channels")
plt.ylabel("Number of WAV files")
plt.title("Audio Channel Distribution")

save_figure(
    "32_audio_channels.png"
)


# ============================================================
# 26. WAVEFORM + MEL-SPECTROGRAM
# ============================================================

print_section(
    "21. WAVEFORM AND MEL-SPECTROGRAM"
)

visual_wavs = sorted(
    WAV_FOLDER.rglob("*.wav")
)

if MAX_AUDIO_FILES is not None:
    visual_wavs = visual_wavs[:MAX_AUDIO_FILES]

visual_wavs = visual_wavs[
    :N_AUDIO_VISUAL_SAMPLES
]

print(
    f"Visualizing: "
    f"{len(visual_wavs)} audio files"
)

for idx, wav_path in enumerate(
    visual_wavs,
    start=1
):

    try:

        y, sr = librosa.load(
            str(wav_path),
            sr=None,
            mono=True
        )

        plt.figure(
            figsize=(12, 4)
        )

        librosa.display.waveshow(
            y,
            sr=sr
        )

        plt.xlabel(
            "Time (seconds)"
        )

        plt.ylabel(
            "Amplitude"
        )

        plt.title(
            f"Waveform - {wav_path.name}"
        )

        save_figure(
            f"26_waveform_sample_{idx:02d}.png"
        )

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=80,
            fmax=min(8000, sr // 2)
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        plt.figure(
            figsize=(12, 5)
        )

        librosa.display.specshow(
            mel_db,
            sr=sr,
            x_axis="time",
            y_axis="mel"
        )

        plt.colorbar(
            format="%+2.0f dB"
        )

        plt.xlabel(
            "Time (seconds)"
        )

        plt.ylabel(
            "Mel frequency"
        )

        plt.title(
            f"Mel-Spectrogram - {wav_path.name}"
        )

        save_figure(
            f"27_melspectrogram_sample_{idx:02d}.png"
        )

    except Exception as e:

        print(
            f"[WARNING] "
            f"{wav_path.name}: {e}"
        )


# ============================================================
# 27. AUDIO QUALITY FLAGS
# ============================================================

print_section(
    "22. AUDIO QUALITY FLAGS"
)

quality_rows = []

for _, row in audio_df.iterrows():

    flags = []

    if row["sample_rate"] != 16000:
        flags.append("non_16k")

    if row["channels"] != 1:
        flags.append("non_mono")

    if row["duration_sec"] < 0.5:
        flags.append("very_short")

    if row["duration_sec"] > 20:
        flags.append("very_long")

    if (
        pd.notna(row["peak_amplitude"])
        and row["peak_amplitude"] >= 0.99
    ):
        flags.append("possible_clipping")

    if (
        pd.notna(row["silence_ratio"])
        and row["silence_ratio"] >= 0.90
    ):
        flags.append("mostly_silent")

    quality_rows.append({
        "file_name": row["file_name"],
        "relative_path": row["relative_path"],
        "flags": "|".join(flags),
        "is_suspect": len(flags) > 0,
    })

audio_quality_df = pd.DataFrame(
    quality_rows
)

save_csv(
    audio_quality_df,
    "33_audio_quality_flags.csv"
)

print(
    audio_quality_df["flags"]
    .value_counts()
    .head(30)
)


# ============================================================
# 28. FINAL QUALITY SUMMARY
# ============================================================

print_section(
    "23. FINAL DATASET QUALITY SUMMARY"
)

quality_summary = []


def add_quality(
    issue,
    count,
    total,
    severity,
    note
):

    quality_summary.append({
        "issue": issue,
        "count": int(count),
        "percentage": (
            count / total
            if total
            else 0
        ),
        "severity": severity,
        "note": note,
    })


if "audio_path" in df.columns:

    add_quality(
        "duplicate_audio_path",
        int(
            df["audio_path"]
            .duplicated()
            .sum()
        ),
        len(df),
        "REVIEW",
        "Kiểm tra duplicate trước khi loại."
    )


if "transcript" in df.columns:

    add_quality(
        "missing_transcript",
        int(
            df["transcript"]
            .isna()
            .sum()
        ),
        len(df),
        "REVIEW",
        "Quan trọng cho ASR/TSD."
    )


if SPEAKER_COL in df.columns:

    add_quality(
        "missing_speaker_id",
        int(
            df[SPEAKER_COL]
            .isna()
            .sum()
        ),
        len(df),
        "IMPORTANT",
        "Kiểm tra speaker-independent split."
    )


add_quality(
    "corrupted_audio",
    len(corrupted_df),
    len(wav_files),
    "CRITICAL",
    "Audio không đọc được."
)


add_quality(
    "non_16k_audio",
    int(
        (audio_df["sample_rate"] != 16000)
        .sum()
    ),
    len(audio_df),
    "REVIEW",
    "Kiểm tra sampling-rate requirement."
)


add_quality(
    "non_mono_audio",
    int(
        (audio_df["channels"] != 1)
        .sum()
    ),
    len(audio_df),
    "REVIEW",
    "Kiểm tra nếu model yêu cầu mono."
)


add_quality(
    "very_short_audio",
    int(
        (audio_df["duration_sec"] < 0.5)
        .sum()
    ),
    len(audio_df),
    "REVIEW",
    "Không xóa tự động."
)


add_quality(
    "very_long_audio",
    int(
        (audio_df["duration_sec"] > 20)
        .sum()
    ),
    len(audio_df),
    "REVIEW",
    "Kiểm tra segment."
)


quality_summary_df = pd.DataFrame(
    quality_summary
)

print(
    quality_summary_df.to_string(
        index=False
    )
)

save_csv(
    quality_summary_df,
    "34_quality_summary.csv"
)


# ============================================================
# 29. FINAL JSON
# ============================================================

summary = {
    "dataset": {
        "csv": str(CSV_PATH),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
    },

    "audio": {
        "wav_folder": str(WAV_FOLDER),
        "total_wav": int(len(wav_files)),
        "readable_wav": int(len(audio_df)),
        "corrupted_wav": int(len(corrupted_df)),
    },

    "quality": quality_summary,

    "audio_signal": {
        "duration_mean_sec": float(
            audio_df["duration_sec"].mean()
        ),
        "duration_median_sec": float(
            audio_df["duration_sec"].median()
        ),
        "duration_min_sec": float(
            audio_df["duration_sec"].min()
        ),
        "duration_max_sec": float(
            audio_df["duration_sec"].max()
        ),
        "sampling_rates": {
            str(k): int(v)
            for k, v in
            audio_df["sample_rate"]
            .value_counts()
            .to_dict()
            .items()
        },
        "channels": {
            str(k): int(v)
            for k, v in
            audio_df["channels"]
            .value_counts()
            .to_dict()
            .items()
        },
    }
}

if "toxicity" in df.columns:
    summary["toxicity"] = {
        str(k): int(v)
        for k, v in
        df["toxicity"]
        .value_counts()
        .to_dict()
        .items()
    }

if "split" in df.columns:
    summary["split"] = {
        str(k): int(v)
        for k, v in
        df["split"]
        .value_counts()
        .to_dict()
        .items()
    }

if "audio_source" in df.columns:
    summary["audio_source"] = {
        str(k): int(v)
        for k, v in
        df["audio_source"]
        .value_counts()
        .to_dict()
        .items()
    }

json_path = (
    OUTPUT_DIR
    / "35_dataset_summary.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )

print(
    f"[SAVED JSON] {json_path}"
)


# ============================================================
# 30. DONE
# ============================================================

print_section(
    "EDA COMPLETED SUCCESSFULLY"
)

print(
    f"Dataset:\n{CSV_PATH}"
)

print(
    f"\nWAV:\n{WAV_FOLDER}"
)

print(
    f"\nOutput:\n{OUTPUT_DIR}"
)

print(
    "\nImportant outputs:"
)

for filename in [
    "16_audio_signal_statistics.csv",
    "20_histogram_*.png",
    "30_boxplot_*.png",
    "30_boxplot_iqr_outlier_report.csv",
    "26_waveform_sample_*.png",
    "27_melspectrogram_sample_*.png",
    "33_audio_quality_flags.csv",
    "34_quality_summary.csv",
    "35_dataset_summary.json",
]:
    print(f"  - {filename}")

print(
    "\nSpeechBrain / k2 không được sử dụng."
)
