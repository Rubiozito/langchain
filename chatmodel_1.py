import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

chat_history = []

system_message = SystemMessage(content="Você é Sofia, uma atendende virtual da empresa socium")
chat_history.append(system_message)


while True:
    query = input("Você: ")
    if query.lower() == "sair":
        break
    chat_history.append(HumanMessage(content=query))

    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))

    print(f"Sofia: {response}")

print("Chat History:")
print(chat_history)

