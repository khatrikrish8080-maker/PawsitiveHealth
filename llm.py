
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_chat_model():

     primary = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        api_key=api_key,
        max_retries=2,
    )
    
    fallback = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        api_key=api_key,
        thinking_level="low",
        max_retries=1,
    )
    
    return primary.with_fallbacks([fallback])
