import json
import random

def create_sample_dataset(input_path, output_path, sample_size=100):
    with open(input_path, "r") as f:
        papers = [json.loads(line) for line in f]

    sampled_papers = random.sample(papers, min(sample_size, len(papers)))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled_papers, f, indent=4)

    print(f"Sample dataset of {len(sampled_papers)} reviews saved to {output_path}")

input_file = "data/split/train.json"
output_file = "data/split/sample_subset_train.json"
create_sample_dataset(input_file, output_file, sample_size=200)
