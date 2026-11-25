
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_chat_model():

     api_key = os.getenv("GOOGLE_API_KEY")

     primary = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        api_key=api_key,
        max_retries=2,
    )
    return primary
