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
load_dotenv()

class JsonSchema(BaseModel):
    sentiment: Literal['Positive', 'Negative', 'Neutral'] = Field(description="The sentiment of the review")
    sentiment_reason: str = Field( description="A concise explanation justifying the detected sentiment based on the review's language, tone, and choice of words")
    tone: Literal['Formal', 'Informal', 'Neutral', 'Supportive', 'Critical', 'Balanced'] = Field(description="The tone of the review")
    tone_reason: str = Field(description="An explanation of the detected tone using specific words, phrases, or stylistic features")

pydantic_parser = PydanticOutputParser(pydantic_object=JsonSchema)

# model = ChatGroq(
#     temperature=0, 
#     model_name="qwen-2.5-32b",
#     api_key=os.getenv('GROQ_API_KEY'))

# model = ChatOpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.getenv("OPENROUTER_API_KEY"),
# )

model = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    api_key=os.getenv('GEMINI_API_KEY')
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

def analyze_tone_and_sentiment(input_file: str, output_file: str):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as inputfile:
        data = json.load(inputfile)

    review = []
    for item in tqdm(data):
        sentiment = []
        sentiment_reason = []
        tone = []
        tone_reason = []
        for content in tqdm(item['review_contents']):
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
                    retries = retries + 1
                    retry_wait_time = retries * 5
                    print(f"Error {e}\n Retrying in {retry_wait_time} seconds...")
                    time.sleep(retry_wait_time)

        item['sentiment'] = sentiment
        item['sentiment_reason'] = sentiment_reason
        item['tone'] = tone
        item['tone_reason'] = tone_reason
        review.append(item)
        with open(output_file, mode='w', encoding='utf-8') as f:
            json.dump(review, f, indent=4)

if __name__ == '__main__':
    analyze_tone_and_sentiment('data/processed/review_extraction/review_extraction.json', 'data/processed/sentiment_analysis/sentiment_analysis_chain.json')
