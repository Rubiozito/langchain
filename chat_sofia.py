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
history = MongoDBChatMessageHistory(connection_string=mongo_uri, session_id="user_chat6", database_name="langchain-socium")

system_message = SystemMessage(
    content="Você é Sofia, uma atendente virtual da empresa Socium."
            "Você deve se apresentar e oferecer ajuda ao usuário."
            "Você pode explicar sobre a empresa ou agendar uma reunião."
            "Pergunte ao usuário o que deseja e classifique a intenção."
)

#Get history

if not history.messages:
    history.add_message(system_message)

def get_recent_chat_history(history, max_messages=10):
    chat_history = history.messages[-max_messages:] 
    formatted_history = "\n".join(
        [f"{'Você' if isinstance(msg, HumanMessage) else 'Sofia'}: {msg.content}" for msg in chat_history]
    )
    return formatted_history

# Prompt templates

# Prompt de identificação da intenção do usuário
identification_template = ChatPromptTemplate.from_messages([
    ("system", "Você é Sofia, uma atendente virtual da empresa Socium. Você possui o histórico de mensagens do usuário, sempre use o histórico para manter o contexto coerente e te ajudar na tomada de decisões."
               "Identifique a intenção do usuário entre estas opções: "
               "'Conhecer a empresa' ou 'Agendar uma reunião'."
               "Responda apenas com a opção mais adequada."),
    ("human", "{user_input}"),
])

# Prompt de apresentação da empresa
presentation_template = ChatPromptTemplate.from_messages([
    ("system", f"Você é Sofia, uma atendente virtual da empresa Socium. Você possui o histórico de mensagens do usuário, sempre use o histórico para manter o contexto coerente e te ajudar na tomada de decisões. você deve apresentar a empresa com base nas seguintes informações: {infos_socium}. "),
    ("human", "{user_input}"),
])


# Prompt de agendamento de reunião
schedule_template = ChatPromptTemplate.from_messages([
    ("system", "Você é Sofia, uma atendente virtual da empresa Socium. Você possui o histórico de mensagens do usuário, sempre use o histórico para manter o contexto coerente e te ajudar na tomada de decisões."
               "Ajude o usuário a agendar uma reunião. Pergunte sobre "
               "disponibilidade de horário e informações de contato."),
    ("human", "{user_input}"),
])

client_questions_template = ChatPromptTemplate.from_messages([
    ("system", '''Você é Sofia, uma atendente virtual da empresa Socium. 
               Seu objetivo é coletar algumas informações do cliente antes de agendar uma reunião. 
               Você deve fazer 3 perguntas sequenciais, garantindo um fluxo natural da conversa.

            <perguntas>
               1. Qual o número aproximado de funcionários da empresa?
               2. Qual a idade da empresa?
               3. A empresa está em processo de transformação tecnológica?
            </perguntas>

                **Regras para a conversa:**
            - Faça **uma pergunta por vez** e aguarde a resposta antes de continuar.
            - **Nunca repita** uma pergunta que já foi respondida corretamente.
            - Se o cliente não responder corretamente ou der uma resposta vaga, reformule a pergunta para obter uma resposta clara.
            - Use o **histórico de mensagens** para lembrar quais perguntas já foram feitas e garantir um fluxo natural da conversa.
            - Se todas as perguntas forem respondidas, finalize essa etapa e prossiga para o agendamento da reunião.
            - Caso o cliente mude de assunto ou tenha dúvidas, responda de maneira natural antes de continuar as perguntas.

                **Objetivo:** 
            - Coletar essas informações de forma natural, sem parecer um questionário mecânico.
            - Adaptar-se ao tom da conversa do usuário, garantindo uma interação fluida.
     '''),

    ("human", "{user_input}"),
])

asking_questions = False

#chat loop
while True:
    user_input = input("Você: ")
    if user_input.lower() == "sair":
        break

    history.add_message(HumanMessage(content=user_input))

    chat_history = get_recent_chat_history(history)

    classification_response = identification_template | model | StrOutputParser()
    classification = classification_response.invoke({"user_input": user_input, "chat_history": chat_history}).strip().lower()

    if asking_questions:
        chain = client_questions_template | model | StrOutputParser()
        response = chain.invoke({"user_input": user_input, "chat_history": chat_history})
        
        if "Todas as perguntas foram respondidas" in response.lower():
            print("\nSofia: Obrigada! Agora podemos continuar com o agendamento.")
            asking_questions = False
            chain = schedule_template | model | StrOutputParser()
        else:
            print(f"Sofia: {response}")
            history.add_message(AIMessage(content=response))
            continue  

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

