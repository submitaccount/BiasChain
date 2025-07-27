from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
import os
import json
from tqdm import tqdm
import streamlit as st
import time
load_dotenv()
from config import get_secret

model = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    # model='gemini-2.0-flash-thinking-exp',
    api_key=get_secret('GEMINI_API_KEY'),
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

def compare_reviews_pairwise(input_data: list):
    paper_title = input_data[0]['paper_title']
    paper_abstract = input_data[0]['paper_abstract']
    reviews = input_data[0]['review_contents']

    progress_bar = st.progress(0, text="Analyzing internal consistency checks...")


    is_consistent_with_others = []
    alignment_score = []
    contradictory_points = []
    possible_bias_flags = []
    summary_of_differences = []

    for i in range(len(reviews)):
        current_review = reviews[i]
        other_reviews = [rev for rev in reviews if rev!=current_review]
        retries, max_retries = 0, 10
        while retries < max_retries:
            try:
                result = pairwise_comparison_chain.invoke(
                        {'paper_title': paper_title,
                         'paper_abstract': paper_abstract,
                         'main_review': current_review,
                         'other_reviews': '\n**Other Review**\n\n'.join(other_reviews)
                         }
                        )
                print(result)
                is_consistent_with_others.append(result.is_consistent_with_others)
                alignment_score.append(result.alignment_score)
                contradictory_points.append(result.contradictory_points)
                possible_bias_flags.append(result.possible_bias_flags)
                summary_of_differences.append(result.summary_of_differences)
                break
            except Exception as e:
                retries += 1
                retry_wait_time = retries * 5
                print(f"Error {e}\n Retrying in {retry_wait_time} seconds...")
                time.sleep(retry_wait_time)

        progress_bar.progress((i + 1) / len(reviews), text=f"Processed {i + 1} of {len(reviews)}")

    return is_consistent_with_others, alignment_score, contradictory_points, possible_bias_flags, summary_of_differences