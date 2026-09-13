import pandas as pd

# Load CSV
df = pd.read_csv(
    "candidate_segments_v1.csv"
)

# Giữ lại các dòng KHÔNG chứa chuỗi này
df = df[
    ~df["segment_filename"]
    .str.contains(
        "eYv-pF4d-nM",
        na=False
    )
]

# Ghi đè lại file
df.to_csv(
    "candidate_segments_v1.csv",
    index=False
)

print(len(df))