# from langchain_groq import ChatGroq # type: ignore
from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json
# from typing import List
from tqdm import tqdm
import time
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
from config import get_secret

class JsonSchema(BaseModel):
    sentiment: Literal['Positive', 'Negative', 'Neutral'] = Field(description="The sentiment of the review")
    sentiment_reason: str = Field( description="A concise explanation justifying the detected sentiment based on the review's language, tone, and choice of words")
    tone: Literal['Formal', 'Informal', 'Neutral', 'Supportive', 'Critical', 'Balanced'] = Field(description="The tone of the review")
    tone_reason: str = Field(description="An explanation of the detected tone using specific words, phrases, or stylistic features")

pydantic_parser = PydanticOutputParser(pydantic_object=JsonSchema)

model = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    api_key=get_secret('GEMINI_API_KEY')
)

sentiment_prompt = PromptTemplate(
    template="""
    You are an expert in analyzing peer review text.
    Analyze the following peer review and classify it according to the following schema:
    {json_schema}

    The review starts here:
    {text}
    """,
    input_variables=['text'],
    partial_variables={'json_schema': pydantic_parser.get_format_instructions()}
)

chain = sentiment_prompt | model | pydantic_parser

def analyze_sentiment_and_tone(input: list)->list:
    sentiment = []
    sentiment_reason = []
    tone = []
    tone_reason = []

    progress_bar = st.progress(0, text="Analyzing sentiment and tone...")
    total = len(input)

    for idx, content in enumerate(input):
        retries, max_retries = 0, 10
        while retries < max_retries:
            try:
                result = chain.invoke({'text': content})
                sentiment.append(result.sentiment)
                sentiment_reason.append(result.sentiment_reason)
                tone.append(result.tone)
                tone_reason.append(result.tone_reason)
                break
            except Exception as e:
                retries += 1
                retry_wait_time = retries * 5
                print(f"Error {e}\n Retrying in {retry_wait_time} seconds...")
                time.sleep(retry_wait_time)

        progress_bar.progress((idx + 1) / total, text=f"Processed {idx + 1} of {total}")
        
    return sentiment, sentiment_reason, tone, tone_reason