
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_chat_model():

    model = ChatGoogleGenerativeAI(  
        model="gemini-3.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    return model

if __name__ == "__main__":
    model=get_chat_model()
    response=model.invoke("Can you name some famous dog breeds?")
    print(response.content)