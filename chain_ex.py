from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_openai import ChatOpenAI
import os

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Você é Sofia, uma atendente virtual da empresa socium, uma empresa que atua nas seguintes áreas:{areas}"),
    ("human", "Quais são as áreas de atuação da empresa?"),
])

uppercase_out = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

chain = prompt_template | model | StrOutputParser() | uppercase_out | count_words

result = chain.invoke({"areas": "- IA - Criação de Infraestrutura - Criação de base de dados - Automações em geral - Construção de aplicativos - Transformação tecnológica - Chatbots - Entre outras coisas.."})

print(result)