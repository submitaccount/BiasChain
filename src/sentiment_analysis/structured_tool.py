import os
import json
import time
import re
from tqdm import tqdm
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from transformers import pipeline
from dotenv import load_dotenv
load_dotenv()
import logging

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/sentiment_analysis.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class JsonSchema(BaseModel):
    sentiment: Literal['Positive', 'Negative', 'Neutral'] = Field(
        description="The sentiment of the review"
    )
    sentiment_reason: str = Field(
        description="A concise explanation justifying the detected sentiment based on the review's language, tone, and choice of words"
    )

pydantic_parser = PydanticOutputParser(pydantic_object=JsonSchema)

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

model_id = "distilbert-base-uncased-finetuned-sst-2-english"

def clean_json_output(raw_output: str) -> str:
    logger.info("Raw output before cleaning: %s", raw_output)
    
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_output, flags=re.DOTALL)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.DOTALL)
    # cleaned = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r"\\\\", cleaned)
    
    logger.debug("Cleaned output: %s", cleaned)
    return cleaned.strip()

def parse_output(output_json):
    list_pred = []
    for i in range(len(output_json[0])):
        label = output_json[0][i]['label']
        score = output_json[0][i]['score']
        list_pred.append((label, score))
    return list_pred

def get_prediction(model_id):
    classifier = pipeline("text-classification", model=model_id, top_k=None)
    def predict(review):
        prediction = classifier(review)
        logger.info("HuggingFace Tool input: %s", review)
        logger.info("HuggingFace Raw prediction: %s", prediction)
        return parse_output(prediction)
    return predict

@tool(
    description="Accepts a text review as input and returns structured sentiment analysis in JSON format (including sentiment and a concise sentiment_reason).",
    return_direct=True
)
def analyze_sentiment(text: str) -> str:
    logger.info("Running `analyze_sentiment` on input length: %d", len(text))
    result = run_structured_chain(text)
    logger.info("Structured result: %s", result)
    return json.dumps(result, indent=4)

predict_sentiment = get_prediction(model_id)

llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-lite', 
    api_key=os.getenv('GEMINI_API_KEY')
)


def run_structured_chain(review_text: str) -> dict:
    # The chain: prompt template -> LLM -> pydantic parser
    logger.info("Formatting prompt for review.")
    formatted_prompt = sentiment_prompt.format(text=review_text)
    logger.info("Sending prompt to LLM.")
    llm_response = llm.invoke(formatted_prompt) 
    logger.info("LLM response: %s", llm_response)
    raw_output = llm_response.content
    logger.info("LLM raw output: %s", raw_output)
    cleaned_output = clean_json_output(raw_output)
    logger.info("Cleaned output: %s", cleaned_output)
    try:
        parsed_output = pydantic_parser.parse(cleaned_output)
        logger.info("Parsed output: %s", parsed_output)
        return parsed_output.model_dump()
    except Exception as e:
        logger.exception("Parsing failed for LLM output: %s", raw_output)
        raise

tools = [analyze_sentiment]
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True
)

def analyze_reviews_sentiment(input_file: str, output_file: str):
    logger.info("Starting sentiment analysis pipeline.")

    if not os.path.exists(input_file):
        logger.error("Input file not found: %s", input_file)
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    with open(input_file, 'r', encoding='utf-8') as inputfile:
        data = json.load(inputfile)

    # Check if the output file exists; if so, load existing results.
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            processed = json.load(f)
        logger.info("Found existing output with %d items processed.", len(processed))
    else:
        processed = []

    review_results = processed if processed else []
    start_index = len(review_results)
    logger.info("Resuming from index %d.", start_index)

    for idx in tqdm(range(start_index, len(data)), desc="Processing items"):
        item = data[idx]
        structured_outputs = []
        for content in tqdm(item.get('review_contents', []), desc="Processing review contents", leave=False):
            retries, max_retries = 0, 10
            while retries < max_retries:
                try:
                    input_instruction = (
                        f"Analyze the sentiment of this review text in a structured format:\n\n{content}"
                    )
                    logger.info("Calling agent for review content (length=%d)...", len(content))
                    result_text = agent.invoke(input_instruction)
                    structured_outputs.append(result_text)
                    logger.info("Agent response: %s", result_text)
                    break
                except Exception as e:
                    retries += 1
                    retry_wait_time = retries * 5
                    logger.warning("Error on retry %d: %s", retries, str(e))
                    time.sleep(retry_wait_time)
            else:
                logger.error("Max retries exceeded for review content.")
                structured_outputs.append("Error analyzing sentiment.")
        item['sentiment_analysis'] = structured_outputs
        review_results.append(item)

        with open(output_file, mode='w', encoding='utf-8') as f:
            json.dump(review_results, f, indent=4)
        logger.info("Processed item %d/%d.", idx + 1, len(data))
    logger.info("Sentiment analysis completed. Output saved to: %s", output_file)

if __name__ == '__main__':
    analyze_reviews_sentiment(
        'data/processed/review_extraction/review_extraction.json',
        'data/processed/sentiment_analysis/sentiment_analysis_via_structured_tool.json'
    )
