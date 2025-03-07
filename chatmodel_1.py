import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pymongo import MongoClient
from langchain_mongodb import MongoDBChatMessageHistory


load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

history = MongoDBChatMessageHistory(connection_string=mongo_uri, session_id="user_chat1", database_name="langchain-socium")

system_message = SystemMessage(content="Você é Sofia, uma atendende virtual da empresa socium")
history.add_message(system_message)



while True:
    query = input("Você: ")
    if query.lower() == "sair":
        break
    history.add_message(HumanMessage(content=query))

    result = model.invoke(history.messages)
    response = result.content
    history.add_message(AIMessage(content=response))

    print(f"Sofia: {response}")

print("Chat History:")
print(history.messages)

