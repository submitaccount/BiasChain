from transformers import pipeline
from langchain.tools import tool
from langchain.agents import initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv() 

def parse_output(output_json):
    list_pred = []
    for i in range(len(output_json[0])):
        label = output_json[0][i]['label']
        score = output_json[0][i]['score']
        list_pred.append((label, score))
    return list_pred

def get_prediction(model_id):
    classifier = pipeline("text-classification", model=model_id, return_all_scores=True)
    
    def predict(review):
        prediction = classifier(review)
        print(prediction)  
        return parse_output(prediction)
    
    return predict

@tool(description="Takes a string as input and returns sentiment predictions (labels with scores).")
def analyze_sentiment(text: str) -> str:
    result = predict_sentiment(text)
    formatted_result = "\n".join([f"{label}: {score:.2f}" for label, score in result])
    return formatted_result

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
predict_sentiment = get_prediction(model_id)

tools = [analyze_sentiment]
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-lite',
    api_key=os.getenv('GEMINI_API_KEY')
)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

if __name__ == "__main__":
    input_text = "I love the service provided by this company!"
    result = agent.run(input_text)
    print("Agent's output:\n", result)
