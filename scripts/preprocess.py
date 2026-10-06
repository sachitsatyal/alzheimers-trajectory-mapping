"""Raw TARCC files -> data/processed/{train,val,test}.npz and meta.json."""
import pandas as pd

from alzheimers_trajectory_mapping.codebook import load_codebook
from alzheimers_trajectory_mapping.sequences import build, save
from alzheimers_trajectory_mapping.transforms import clean

df = pd.read_excel("data/raw/TARCC_Main_Data.xlsx")
fm = pd.read_csv("configs/feature_map.csv")
cb = load_codebook("data/raw/TARCC_Codebook.xlsx", df.columns)
arrays, meta = build(clean(df, cb, fm), seed=42)
save(arrays, meta, "data/processed")
print({split: len(a["PATID"]) for split, a in arrays.items()})