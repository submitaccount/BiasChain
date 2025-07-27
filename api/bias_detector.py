from prompts import (child_parser, parent_parser,
                    novelty_bias_prompt, methodology_bias_prompt, confirmation_bias_prompt, positive_results_bias_prompt, linguistic_bias_prompt,
                    parent_agent_prompt, output_parser_prompt)
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.tools import tool # Type : ignore
from dotenv import load_dotenv
from tqdm import tqdm
import json
import time
import os
import warnings
import streamlit as st
# import re # type : ignore
load_dotenv()
warnings.filterwarnings(action='ignore')
from config import get_secret


parent_llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    temperature=0.5,
    api_key=get_secret('GEMINI_API_KEY')
)

tools_llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-thinking-exp',
    temperature=0.25,
    api_key=get_secret('GEMINI_API_KEY')
)

parsing_llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-lite',
    temperature=0,
    api_key=get_secret('GEMINI_API_KEY')
)

def novelty_bias_detector_function(input: str)-> json:
    print("Entered Novelty tool")
    retries, max_retries = 0, 5
    while retries<max_retries:
        try:
            chain = novelty_bias_prompt | tools_llm | child_parser
            response = chain.invoke({"input": input})
            return response
        except Exception as e:
            retries = retries + 1
            print(f"Error {e} in topic_bias_detector_function, retrying in {5 * retries} sec...")
            time.sleep(5 * retries)

def confirmation_bias_detector_function(input: str)-> json:
    print("Entered Confirmation tool")
    retries, max_retries = 0, 5
    while retries<max_retries:
        try:
            chain = confirmation_bias_prompt | tools_llm | child_parser
            response = chain.invoke({"input": input})
            return response
        except Exception as e:
            retries = retries + 1
            print(f"Error {e} in confirmation_bias_detector_function, retrying in {5 * retries} sec...")
            time.sleep(5 * retries)


def methodology_bias_detector_function(input: str)-> json:
    print("Entered Methodology tool")
    retries, max_retries = 0, 5
    while retries<max_retries:
        try:
            chain = methodology_bias_prompt | tools_llm | child_parser
            response = chain.invoke({"input": input})
            return response
        except Exception as e:
            retries = retries + 1
            print(f"Error {e} in content_bias_detector_function, retrying in {5 * retries} sec...")
            time.sleep(5 * retries)

def positive_result_bias_detector_function(input: str)-> json:
    print("Entered Positive Result tool")
    retries, max_retries = 0, 5
    while retries<max_retries:
        try:
            chain = positive_results_bias_prompt | tools_llm | child_parser
            response = chain.invoke({"input": input})
            return response
        except Exception as e:
            retries = retries + 1
            print(f"Error {e} in prior_knowledge_bias_detector_function, retrying in {5 * retries} sec...")
            time.sleep(5 * retries)

def linguistic_bias_detector_function(input: str)-> json:
    print("Entered Lingustic tool")
    retries, max_retries = 0, 5
    while retries<max_retries:
        try:
            chain = linguistic_bias_prompt | tools_llm | child_parser
            response = chain.invoke({"input": input})
            return response
        except Exception as e:
            retries = retries + 1
            print(f"Error {e} in prior_knowledge_bias_detector_function, retrying in {5 * retries} sec...")
            time.sleep(5 * retries)

def get_final_formatted_output(input: any)-> json:
    print("Entered formatting tool")
    retries, max_retries = 0, 5
    while retries<max_retries:
        try:
            chain = output_parser_prompt | parsing_llm | parent_parser
            response = chain.invoke({"input": input})
            return response
        except Exception as e:
            retries = retries + 1
            print(f"Error {e} in get_final_formatted_output, retrying in {5 * retries} sec...")
            time.sleep(5 * retries)


novelty_bias_detector = Tool(
    name="NoveltyBiasDetector", 
    func=novelty_bias_detector_function, 
    description="Detects if there is NOVELTY BIAS present in review",
    )

confirmation_bias_detector = Tool(
    name='ConfirmationBiasDetector', 
    func=confirmation_bias_detector_function,
    description="Detects if there is CONFIRMATION BIAS present in review"
    )

methodology_bias_detector = Tool(
    name='MethodologyBiasDetector', 
    func=methodology_bias_detector_function,
    description="Detects if there is METHODOLOGY BIAS present in review"
    )

positive_result_bias_detector = Tool(
    name='PositiveResultsBiasDetector',
    func=positive_result_bias_detector_function,
    description="Detects if there is POSITIVE RESULT BIAS present in review"
    )

linguistic_bias_detector = Tool(
    name='LinguisticBiasDetector',
    func=linguistic_bias_detector_function,
    description="Detects if there is LINGUISTIC/WRITING STYLE BIAS present in review"
    )


tools = [novelty_bias_detector, 
         confirmation_bias_detector, 
         methodology_bias_detector, 
         positive_result_bias_detector, 
         linguistic_bias_detector]

agent = initialize_agent(
    llm=parent_llm, 
    tools=tools, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    handle_parsing_errors=True
)


def detect_bias(input_data: list):
    paper_title = input_data[0]['paper_title']
    paper_abstract = input_data[0]['paper_abstract']
    reviews = input_data[0]['review_contents']

    progress_bar = st.progress(0, text="Analyzing bias checks...")

    bias_detected = []
    bias_type = []
    confindence_score = []
    evidence = []
    suggestion_for_improvements = []

    for i in range(len(reviews)):
        original_review = reviews[i]
        # other_reviews = [rev for rev in reviews if rev!=original_review]
        tone = input_data[0]['tone'][i]
        tone_reason = input_data[0]['tone_reason'][i]
        consistency = input_data[0]['consistency'][i]
        consistency_reason = input_data[0]['consistency_reason'][i]
        is_consistent_with_others = input_data[0]['is_consistent_with_others'][i]
        alignment_score = input_data[0]['alignment_score'][i]
        contradictory_points = input_data[0]['contradictory_points'][i]
        possible_bias_flags = input_data[0]['possible_bias_flags'][i]
        summary_of_differences = input_data[0]['summary_of_differences'][i]

        retries, max_retries = 0, 10
        while retries < max_retries:
            try:
                structured_prompt = parent_agent_prompt.invoke(
                    {
                        "paper_title": paper_title,
                        "paper_abstract": paper_abstract,
                        "original_review": original_review,
                        "tone": tone,
                        "tone_reason": tone_reason,
                        "consistency": consistency,
                        "consistency_reason": consistency_reason,
                        "is_consistent_with_others": is_consistent_with_others,
                        "alignment_score": alignment_score,
                        "contradictory_points": contradictory_points,
                        "possible_bias_flags": possible_bias_flags,
                        "summary_of_differences": summary_of_differences,
                    },
                )

                result = agent.invoke(structured_prompt.text)

                final_jsonified_result = get_final_formatted_output(result['output'])

                bias_detected.append(final_jsonified_result.bias_detected)
                bias_type.append(final_jsonified_result.bias_type)
                confindence_score.append(final_jsonified_result.confindence_score)
                evidence.append(final_jsonified_result.evidence)
                suggestion_for_improvements.append(final_jsonified_result.suggestion_for_improvements)
                break

            except Exception as e:
                retries = retries + 1
                print(f"Error!! {e}")
                print(f"Retrying in {5*retries} seconds....")
                time.sleep(5*retries)
            
        progress_bar.progress((i + 1) / len(reviews), text=f"Processed {i + 1} of {len(reviews)}")

    return bias_detected, bias_type, confindence_score, evidence, suggestion_for_improvements

