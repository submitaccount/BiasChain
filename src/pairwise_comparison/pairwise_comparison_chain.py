from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
import os
import json
from tqdm import tqdm
import time
load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    # model='gemini-2.0-flash-thinking-exp',
    api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0
)

class JsonSchema(BaseModel):
    is_consistent_with_others: Literal['True', 'False'] = Field(description="Check whether the current review is consistent with other reviews")
    alignment_score: float = Field(ge=0, le=10, description="Alignment score of the current review with other reviews")
    contradictory_points: str = Field(description="List down contradictory points between the current review and other reviews, if any, else write 'No contradictory points'")
    possible_bias_flags: str = Field(description="List down the reasons for the current review being biased, if any, else write 'No possible bias flags'")
    summary_of_differences: str = Field(description="List down the summary of differences between current review and other reviews, if any, else write 'No differences found'")
parser = PydanticOutputParser(pydantic_object=JsonSchema)

prompt = PromptTemplate(
    template="""
    You are an expert meta-reviewer.
    You are provided with paper title, abstract and two sets of reviews.
    One set contains only one review and other set contains the list of reviews.
    Your main goal is to compare this one review with list of the other reviews and give the output in following schema:
    {json_schema}

    Here is the data:
    Paper Title: {paper_title}
    Paper Abstract: {paper_abstract}
    One Review: {main_review}
    Other Reviews: {other_reviews}
    """,
    input_variables=['paper_title', 'paper_abstract', 'main_review', 'other_reviews'],
    partial_variables={'json_schema': parser.get_format_instructions()}
)

pairwise_comparison_chain = prompt | model | parser

def combinatorial_review_comparison(input_file: str, output_file: str):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as inputfile:
        data = json.load(inputfile)

    try:
        with open(output_file, 'r', encoding='utf-8') as outfile:
            processed_reviews = json.load(outfile)
    except:
        processed_reviews = []

    paper_ids_processed = [item['paper_id'] for item in processed_reviews]

    for item in tqdm(data):
        if item['paper_id'] in paper_ids_processed:
            print(f"{item['paper_id']} already processed, skipping...")
            continue

        review_comparison = []
        
        for idx, review in enumerate(tqdm(item['review_contents'])):
            current_review_id = item['review_ids'][idx]
            current_review = item['review_contents'][idx]
            other_reviews = [rev for rev in item['review_contents'] if rev!=current_review]

            retries, max_retries = 0, 10
            while retries < max_retries:
                try:
                    result = pairwise_comparison_chain.invoke(
                        {'paper_title': item['paper_title'],
                         'paper_abstract': item['paper_abstract'],
                         'main_review': current_review,
                         'other_reviews': '\n**Other Review**\n\n'.join(other_reviews)
                         }
                        )
                    break
                except Exception as e:
                    retries = retries + 1
                    retry_wait_time = retries * 5
                    print(f"Error! Retrying in {retry_wait_time} seconds...")
                    time.sleep(retry_wait_time)

            pairwise_comparison = {
                'review_id': current_review_id,
                'comparison': dict(result)
            }

            review_comparison.append(dict(pairwise_comparison))
            
        item['pairwise_comparison'] = review_comparison
        processed_reviews.append(item)
        with open(output_file, 'w', encoding='utf-8') as dump_file:
            json.dump(processed_reviews, dump_file, indent=4)

if __name__ == '__main__':
    combinatorial_review_comparison('data/processed/consistency_check/consistency_check_flash_thinking.json', 'data/processed/pairwise_comparison/pairwise_comparison_flash_thinking.json')