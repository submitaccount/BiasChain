from prompts import (child_parser, parent_parser,
                    topic_bias_prompt, confirmation_bias_prompt, content_bias_prompt, prior_knowledge_bias_prompt,
                    parent_agent_prompt, output_parser_prompt)
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from dotenv import load_dotenv
from tqdm import tqdm
import json
import time
import os
# import re # type : ignore
load_dotenv()

parent_llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    temperature=0.5,
    api_key=os.getenv('GEMINI_API_KEY')
)

tools_llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-thinking-exp',
    temperature=0.25,
    api_key=os.getenv('GEMINI_API_KEY')
)

def topic_bias_detector_function(input: str)-> json:
    retries, max_retries = 0, 5
    while retries<max_retries:
        try:
            chain = topic_bias_prompt | tools_llm | child_parser
            response = chain.invoke({"input": input})
            return response
        except Exception as e:
            retries = retries + 1
            print(f"Error {e} in topic_bias_detector_function, retrying in {5 * retries} sec...")
            time.sleep(5 * retries)

def confirmation_bias_detector_function(input: str)-> json:
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


def content_bias_detector_function(input: str)-> json:
    retries, max_retries = 0, 5
    while retries<max_retries:
        try:
            chain = content_bias_prompt | tools_llm | child_parser
            response = chain.invoke({"input": input})
            return response
        except Exception as e:
            retries = retries + 1
            print(f"Error {e} in content_bias_detector_function, retrying in {5 * retries} sec...")
            time.sleep(5 * retries)

def prior_knowledge_bias_detector_function(input: str)-> json:
    retries, max_retries = 0, 5
    while retries<max_retries:
        try:
            chain = prior_knowledge_bias_prompt | tools_llm | child_parser
            response = chain.invoke({"input": input})
            return response
        except Exception as e:
            retries = retries + 1
            print(f"Error {e} in prior_knowledge_bias_detector_function, retrying in {5 * retries} sec...")
            time.sleep(5 * retries)

def get_final_formatted_output(input: any)-> json:
    retries, max_retries = 0, 5
    while retries<max_retries:
        try:
            chain = output_parser_prompt | parent_llm | parent_parser
            response = chain.invoke({"input": input})
            return response
        except Exception as e:
            retries = retries + 1
            print(f"Error {e} in get_final_formatted_output, retrying in {5 * retries} sec...")
            time.sleep(5 * retries)


topic_bias_detector = Tool(
    name="TopicBiasDetector", 
    func=topic_bias_detector_function, 
    description="Detects if there is TOPIC/METHODOLOGY BIAS present in review",
    )

confirmation_bias_detector = Tool(
    name='ConfirmationBiasDetector', 
    func=confirmation_bias_detector_function,
    description="Detects if there is CONFIRMATION BIAS present in review"
    )

content_bias_detector = Tool(
    name='ContentBiasDetector', 
    func=content_bias_detector_function,
    description="Detects if there is WRITING STYLE/CONTENT BIAS present in review"
    )

prior_knowledge_bias_detector = Tool(
    name='PriorKnowledgeDetector',
    func=prior_knowledge_bias_detector_function,
    description="Detects if there is PRIOR KNOWLEDGE BIAS present in review"
    )


tools = [topic_bias_detector, confirmation_bias_detector, content_bias_detector, prior_knowledge_bias_detector]

agent = initialize_agent(
    llm=parent_llm, 
    tools=tools, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True
)


def run_bias_detection_chain(input_file: str, output_file: str):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file ({input_file}) not found")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs('temp', exist_ok=True)

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

        paper_title = item['paper_id']
        paper_abstract = item['paper_abstract']

        bias_output = []

        for idx, review_id in enumerate(tqdm(item['review_ids'])):
            original_review = item['review_contents'][idx]
            tone = item['tone'][idx]
            tone_reason = item['tone_reason'][idx]
            consistency = item['consistency'][idx]
            consistency_reason = item['consistency_reason'][idx]
            is_consistent_with_others = [item['pairwise_comparison'][idx]['comparison']['is_consistent_with_others'] if review_id==item['pairwise_comparison'][idx]['review_id'] else "BAD_VALUE"][0]
            alignment_score = item['pairwise_comparison'][idx]['comparison']['alignment_score']
            contradictory_points = item['pairwise_comparison'][idx]['comparison']['contradictory_points']
            possible_bias_flags = item['pairwise_comparison'][idx]['comparison']['possible_bias_flags']
            summary_of_differences = item['pairwise_comparison'][idx]['comparison']['summary_of_differences']

            if is_consistent_with_others == "BAD_VALUE":
                raise ValueError(f"BAD VALUE ==> Review Ids didn't match as expected!!")
                
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

                    with open('temp/structured_result.txt', 'a', encoding='utf-8') as structured_op_file:
                        structured_op_file.write(structured_prompt.text)

                    result = agent.invoke(structured_prompt.text, config={"handle_parsing_errors": "True"})

                    final_jsonified_result = get_final_formatted_output(result['output'])

                    bias_output.append({
                        "review_id": review_id,
                        "bias_detection_output": dict(final_jsonified_result)
                    })

                    break             
                except Exception as e:
                    retries = retries + 1
                    print(f"Error!! {e}")
                    print(f"Retrying in {5*retries} seconds....")
                    time.sleep(5*retries)

            item['bias_detection_chain_output'] = bias_output
            processed_reviews.append(item)
            time.sleep(5)

        with open(output_file, 'w', encoding='utf-8') as dump_file:
            json.dump(processed_reviews, dump_file, indent=4)
        time.sleep(10)


if __name__ == '__main__':
    run_bias_detection_chain('data/processed/pairwise_comparison/pairwise_comparison_flash_thinking.json', 'data/processed/bias_detection/bias_detection_tools_thinking.json')  