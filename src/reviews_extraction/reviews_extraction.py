import json
import os
from tqdm import tqdm

def extract_official_reviews(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    extracted_data = []
    for item in tqdm(data):
        review_ids = item.get("review_ids", [])
        review_writers = item.get("review_writers", [])
        review_contents = item.get("review_contents", [])

        filtered_ids = []
        filtered_contents = []
        filtered_writers = []

        for rid, writer, content in zip(review_ids, review_writers, review_contents):
            if writer == "official_reviewer":
                filtered_ids.append(rid)
                filtered_writers.append(writer)
                filtered_contents.append(content)

        if filtered_ids:
            extracted_data.append({
                "paper_id": item.get("paper_id"),
                "paper_title": item.get("paper_title"),
                "paper_abstract": item.get("paper_abstract"),
                "review_ids": filtered_ids,
                "review_writers": filtered_writers,
                "review_contents": filtered_contents
            })

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(extracted_data, outfile, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    extract_official_reviews(
        "data/split/sample_subset_train.json",
        "data/processed/review_extraction/review_extraction.json"
    )

