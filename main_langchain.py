from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=1,
    api_key=os.getenv("OPENAI_API_KEY"))

resposta = llm.invoke("apenas um teste")
print(resposta)