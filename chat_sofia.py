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

model = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

mongo_uri = os.getenv("MONGO_URI")
history = MongoDBChatMessageHistory(connection_string=mongo_uri, session_id="user_chat12", database_name="langchain-socium")

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
                "Aqui está o histórico recente da conversa:\n\n"
               "{chat_history}\n\n"
               "Identifique a intenção do usuário entre estas opções: "
               "'Conhecer a empresa' ou 'Agendar uma reunião'."
               "Responda apenas com a opção mais adequada."),
    ("human", "{user_input}"),
])

# Prompt de apresentação da empresa
presentation_template = ChatPromptTemplate.from_messages([
    ("system", f"Você é Sofia, uma atendente virtual da empresa Socium. Você possui o histórico de mensagens do usuário, sempre use o histórico para manter o contexto coerente e te ajudar na tomada de decisões. você deve apresentar a empresa com base nas seguintes informações: {infos_socium}. "
     "Aqui está o histórico recente da conversa:\n\n"
               "{chat_history}\n\n"),
    ("human", "{user_input}"),
])


# Prompt de agendamento de reunião
schedule_template = ChatPromptTemplate.from_messages([
    ("system", "Você é Sofia, uma atendente virtual da empresa Socium. Você possui o histórico de mensagens do usuário, sempre use o histórico para manter o contexto coerente e te ajudar na tomada de decisões."
            "Aqui está o histórico recente da conversa:\n\n"
            "{chat_history}\n\n"
            "Ajude o usuário a agendar uma reunião. Pergunte sobre "
            "disponibilidade de horário e informações de contato."),
    ("human", "{user_input}"),
])

client_questions_template = ChatPromptTemplate.from_messages([
    ("system", '''Você é Sofia, uma atendente virtual da empresa Socium.
            Aqui está o histórico recente da conversa:\n\n
               {chat_history}\n\n"
               Seu objetivo é coletar algumas informações do cliente antes de agendar uma reunião. 
               Você deve fazer 3 perguntas sequenciais, garantindo um fluxo natural da conversa.

            <perguntas>
               1. Qual o número aproximado de funcionários da empresa?
               2. Qual a idade da empresa?
               3. A empresa está em processo de transformação tecnológica?
            </perguntas>

                **Regras para a conversa:**
            - **Sempre** confira o histórico e veja quais perguntas já foram feitas.
            - Faça **uma pergunta por vez** e aguarde a resposta antes de continuar.
            - **Nunca repita** uma pergunta que já foi respondida corretamente.
            - Se o cliente não responder corretamente ou der uma resposta vaga, reformule a pergunta para obter uma resposta clara.
            - Use o **histórico de mensagens** para lembrar quais perguntas já foram feitas e garantir um fluxo natural da conversa.
            - Se todas as perguntas forem respondidas, não faça mais perguntas e avise o cliente que você tem todas as informações necessárias.
            - Caso o cliente mude de assunto ou tenha dúvidas, responda de maneira natural antes de continuar as perguntas.
            - Caso o usuário responda a pergunta com apenas um número, leve esse número como resposta correta.

                **Objetivo:** 
            - Coletar essas informações de forma natural, sem parecer um questionário mecânico.
            - Adaptar-se ao tom da conversa do usuário, garantindo uma interação fluida.
            
            Após coletar as informações, responda apenas "OK" para continuar com o agendamento da reunião.
     '''),

    ("human", "{user_input}"),
])

questions = [
    "Qual o número aproximado de funcionários da empresa?",
    "Qual a idade da empresa?",
    "A empresa está em processo de transformação tecnológica?"
]

questions_index = 0

asking_questions = False
questions_answered = False


#chat loop
while True:
    user_input = input("Você: ")
    if user_input.lower() == "sair":
        break

    history.add_message(HumanMessage(content=user_input))
    chat_history = get_recent_chat_history(history)

    if asking_questions:
        if questions_index >= len(questions):
            print("\nSofia: Obrigada! Já coletei todas as informações necessárias")
            asking_questions = False
            questions_answered = True
            continue
        prompt_text = f"""Você é Sofia, uma atendente virtual da empresa Socium.
Aqui está o histórico recente da conversa:
{chat_history}

Sua próxima pergunta é:
{questions[questions_index]}

Regras:
- Faça somente uma pergunta por vez e aguarde a resposta.
- Não repita perguntas já respondidas.
- Se a resposta não estiver clara, reformule a pergunta para obter uma resposta precisa.
Após a coleta de todas as informações, responda apenas "OK" para prosseguir.
"""
        chain = RunnableLambda(lambda x: prompt_text) | model | StrOutputParser()
        response = chain.invoke({"user_input": user_input, "chat_history": chat_history})

        if response.strip().lower() == "ok":
            print("\nSofia: Obrigada! Agora podemos continuar com o agendamento.")
            asking_questions = False
            questions_answered = True
        else:
            print(f"Sofia: {response}")
            history.add_message(AIMessage(content=response))
            indice_pergunta += 1
            continue  
        
    else:
        classification_response = identification_template | model | StrOutputParser()
        classification = classification_response.invoke({"user_input": user_input, "chat_history": chat_history}).strip().lower()

        if "conhecer a empresa" in classification:
            chain = presentation_template | model | StrOutputParser()
        elif "agendar uma reunião" in classification:
            asking_questions = True  
            chain = client_questions_template | model | StrOutputParser()
        else:
            chain = RunnableLambda(lambda x: "Desculpe, não entendi sua solicitação. Você deseja conhecer a empresa ou agendar uma reunião?")
    response = chain.invoke({"user_input": user_input, "chat_history": chat_history})
    print(f"Sofia: {response}")


print("\nChat History:")
for msg in history.messages:
    role = "Você" if isinstance(msg, HumanMessage) else "Sofia"
    print(f"{role}: {msg.content}")


