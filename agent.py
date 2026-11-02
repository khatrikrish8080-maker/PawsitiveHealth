import os
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool


# pyrefly: ignore[missing-import]
from rag_engine import retrieve
# pyrefly: ignore[missing-import]
from llm import get_chat_model

load_dotenv()

@tool
def medical_rag_tool(query: str)-> str:
    """
    Use this tool when you need to answer medical questions about pets based on the available documents
    """
    return retrieve(query,k=2)

search=DuckDuckGoSearchRun()

@tool
def location_tool(query):
    """
    Use this tool to find nearby pet clinics,veternary doctors,pet emergency hospitals and addresses based on user location
    """
    result=search.run(query)
    return result

tools=[medical_rag_tool,location_tool]

template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

agent = create_react_agent(
    llm=get_chat_model(),
    tools=tools,
    prompt=prompt
    )


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

def ask_pawsitive_agent(user_query: str) -> str:
    
    response = agent_executor.invoke({"input": user_query})
    
    return response["output"]

if __name__ == "__main__":
    test_query = "Rohan's dog ate chocolate in New Delhi and is vomiting. What should I do and where is a nearby vet?"
    print(ask_pawsitive_agent(test_query))
