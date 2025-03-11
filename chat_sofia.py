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
history = MongoDBChatMessageHistory(connection_string=mongo_uri, session_id="user_chat4", database_name="langchain-socium")

system_message = SystemMessage(
    content="Você é Sofia, uma atendente virtual da empresa Socium. "
            "Você deve se apresentar e oferecer ajuda ao usuário. "
            "Você pode explicar sobre a empresa ou agendar uma reunião. "
            "Pergunte ao usuário o que deseja e classifique a intenção."
)

if not history.messages:
    history.add_message(system_message)

def get_recent_chat_history(history, max_messages=10):
    """Retorna as últimas mensagens do histórico formatadas como contexto."""
    chat_history = history.messages[-max_messages:]  # Limita ao número máximo de mensagens
    formatted_history = "\n".join(
        [f"{'Você' if isinstance(msg, HumanMessage) else 'Sofia'}: {msg.content}" for msg in chat_history]
    )
    return formatted_history

identification_template = ChatPromptTemplate.from_messages([
    ("system", "Você é Sofia, uma atendente virtual da empresa Socium. "
               "Identifique a intenção do usuário entre estas opções: "
               "'Conhecer a empresa' ou 'Agendar uma reunião'. "
               "Responda apenas com a opção mais adequada."),
    ("human", "{user_input}"),
])

presentation_template = ChatPromptTemplate.from_messages([
    ("system", f"Você é Sofia, uma atendente virtual da empresa Socium. você deve apresentar a empresa com base nas seguintes informações: {infos_socium}."),
    ("human", "{user_input}"),
])


schedule_template = ChatPromptTemplate.from_messages([
    ("system", "Você é Sofia, uma atendente virtual da empresa Socium. "
               "Ajude o usuário a agendar uma reunião. Pergunte sobre "
               "disponibilidade de horário e informações de contato."),
    ("human", "{user_input}"),
])

while True:
    user_input = input("Você: ")
    if user_input.lower() == "sair":
        break

    history.add_message(HumanMessage(content=user_input))

    chat_history = get_recent_chat_history(history)

    classification_response = identification_template | model | StrOutputParser()
    classification = classification_response.invoke({"user_input": user_input, "chat_history": chat_history}).strip().lower()

    if "conhecer a empresa" in classification:
        chain = presentation_template | model | StrOutputParser() 
    elif "agendar uma reunião" in classification:
        chain = schedule_template | model | StrOutputParser() 
    else:
        chain = RunnableLambda(lambda x: "Desculpe, não entendi sua solicitação. Você deseja conhecer a empresa ou agendar uma reunião?")

    response = chain.invoke({"user_input": user_input, "chat_history": chat_history})
    history.add_message(AIMessage(content=response))

    print(f"Sofia: {response}")


print("\nChat History:")
for msg in history.messages:
    role = "Você" if isinstance(msg, HumanMessage) else "Sofia"
    print(f"{role}: {msg.content}")

