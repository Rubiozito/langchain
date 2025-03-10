import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mongodb import MongoDBChatMessageHistory
from info.info_socium import infos_socium

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

mongo_uri = os.getenv("MONGO_URI")
history = MongoDBChatMessageHistory(connection_string=mongo_uri, session_id="user_chat1", database_name="langchain-socium")

system_message = SystemMessage(content=f"Você é Sofia, uma atendente virtual da empresa Socium, que atua nas seguintes áreas: - IA - Criação de Infraestrutura - Criação de base de dados - Automações em geral - Construção de aplicativos - Transformação tecnológica - Chatbots - Entre outras coisas..")

if not history.messages:
    history.add_message(system_message)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", f"Você é Sofia, uma atendente virtual da empresa Socium. Áreas de atuação: - IA - Criação de Infraestrutura - Criação de base de dados - Automações em geral - Construção de aplicativos - Transformação tecnológica - Chatbots - Entre outras coisas.."),
    ("human", "{user_input}"),
])

count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")  

chain = prompt_template | model | StrOutputParser() | count_words

while True:
    user_input = input("Você: ")
    if user_input.lower() == "sair":
        break

    history.add_message(HumanMessage(content=user_input))

    response = chain.invoke({"user_input": user_input})

    history.add_message(AIMessage(content=response))

    print(f"Sofia: {response}")

print("\nChat History:")
for msg in history.messages:
    role = "Você" if isinstance(msg, HumanMessage) else "Sofia"
    print(f"{role}: {msg.content}")

