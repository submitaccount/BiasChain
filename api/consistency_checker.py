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
import streamlit as st
load_dotenv()
from config import get_secret

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.0-flash-thinking-exp',
    api_key=get_secret('GEMINI_API_KEY'),
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

def analyze_review_consistency(input: list)->list:
    consistency = []
    consistency_reason = []

    progress_bar = st.progress(0, text="Analyzing internal consistency checks...")
    total = len(input)

    for idx, content in enumerate(input):
        retries, max_retries = 0, 10
        while retries < max_retries:
            try:
                result = consistency_chain.invoke({'review': content})
                consistency.append(result.consistency)
                consistency_reason.append(result.consistency_reason)
                break
            except Exception as e:
                retries += 1
                retry_wait_time = retries * 5
                print(f"Error {e}\n Retrying in {retry_wait_time} seconds...")
                time.sleep(retry_wait_time)

        progress_bar.progress((idx + 1) / total, text=f"Processed {idx + 1} of {total}")
        
    return consistency, consistency_reason