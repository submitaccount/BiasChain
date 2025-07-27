import streamlit as st
import os
from dotenv import load_dotenv

try:
    streamlit_secrets = st.secrets._secrets  
    running_in_streamlit = bool(streamlit_secrets)
except Exception:
    running_in_streamlit = False  

if not running_in_streamlit:
    load_dotenv()

def get_secret(key, default=None):
    """Retrieve secret from Streamlit secrets or environment variables."""
    if running_in_streamlit:
        return st.secrets.get(key, default)
    return os.getenv(key, default)

if __name__ == '__main__':
    print(f"GEMINI API KEY: {get_secret('GEMINI_API_KEY', 'Not Found')}")
