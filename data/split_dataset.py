import os
import datasets

RAW_DATA_PATH = "data/raw/"
SPLIT_DATA_PATH = "data/split/"

def download_peersum():
    print("Downloading PeerSum dataset...")
    peersum_all = datasets.load_dataset("oaimli/PeerSum", split="all")

    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    peersum_all.save_to_disk(RAW_DATA_PATH)
    print(f"Dataset downloaded and saved at {RAW_DATA_PATH}")


def split_peersum():
    print("Loading dataset from disk...")

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError("Dataset not found. Run download_peersum() first.")

    os.makedirs(SPLIT_DATA_PATH, exist_ok=True)

    dataset = datasets.load_from_disk(RAW_DATA_PATH)

    for split_name in ["train", "val", "test"]:
        split_data = dataset.filter(lambda s: s['label'] == split_name)

        split_path = os.path.join(SPLIT_DATA_PATH, split_name)
        split_data.save_to_disk(split_path) 

        json_path = os.path.join(SPLIT_DATA_PATH, f"{split_name}.json")
        split_data.to_json(json_path) 

        print(f"Saved {split_name} split to: {split_path} (Arrow) & {json_path} (JSON)")

if __name__ == "__main__":
    download_peersum()
    split_peersum()
