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
    model = 'gemini-2.0-flash-thinking-exp',
    api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0
)

class JsonSchema(BaseModel):
    consistency: Literal['Yes', 'No'] = Field(description="Is the review consistent in itself")
    consistency_reason: str = Field(description="The reason for the classified consistency")
parser = PydanticOutputParser(pydantic_object=JsonSchema)

prompt = PromptTemplate(
    template="""
    You are an expert in analyzing peer review texts.
    This time you have to check for the consistency of the review.
    Also, check whether the review contradicts itself.
    If no proper review, tell that.
    Classify according to the following schema:
    {json_schema}

    The review starts here:
    {review}
    """,
    input_variables=['review'],
    partial_variables={'json_schema': parser.get_format_instructions()}
)

consistency_chain = prompt | model | parser

def analyze_consistency(input_file: str, output_file: str):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as inputfile:
        data = json.load(inputfile)
    
    try:
        with open(output_file, 'r', encoding='utf-8') as outfile:
            review = json.load(outfile)
    except:
        review = []

    paper_ids_processed = [item['paper_id'] for item in review]
    for item in tqdm(data):
        if item['paper_id'] in paper_ids_processed:
            print(f"{item['paper_id']} already processed, skipping...")
            continue
        consistency = []
        consistency_reason = []
        for content in tqdm(item['review_contents']):
            retries, max_retries = 0, 10
            while retries < max_retries:
                try:
                    result = consistency_chain.invoke({'review': content})
                    consistency.append(result.consistency)
                    consistency_reason.append(result.consistency_reason)
                    break
                except Exception as e:
                    retries = retries + 1
                    retry_wait_time = retries * 5
                    print(f"Error {e}\n Retrying in {retry_wait_time} seconds...")
                    time.sleep(retry_wait_time)

        item['consistency'] =  consistency
        item['consistency_reason'] = consistency_reason
        review.append(item)
        with open(output_file, mode='w', encoding='utf-8') as f:
            json.dump(review, f, indent=4)

if __name__ == '__main__':
    analyze_consistency('data/processed/sentiment_analysis/sentiment_analysis_chain.json', 'data/processed/consistency_check/consistency_check_flash_thinking.json')
