from datasets import load_dataset

# Download directly as Parquet via Hugging Face
dataset = load_dataset("bbfizp/AMLSim-HI-Small", split="train")

#  Save locally directly as Parquet
dataset.to_parquet("/Data/raw/AMLSim-HI-Small.parquet")
