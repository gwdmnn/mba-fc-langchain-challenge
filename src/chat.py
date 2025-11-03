import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from search import search_documents
from langchain_core.runnables import RunnableLambda

load_dotenv()

def retrieve_context(user_question: str) -> dict:
    results = search_documents(user_question)
    context = "\n".join([doc.page_content for doc, _ in results])
    return {
        "database_result": context,
        "user_input": user_question
    }

base_template = PromptTemplate(
    input_variables=["database_result", "user_input"],
    template="""
    CONTEXTO:
    {database_result}

    REGRAS:
    - Responda somente com base no CONTEXTO.
    - Se a informação não estiver explicitamente no CONTEXTO, responda:
    "Não tenho informações necessárias para responder sua pergunta."
    - Nunca invente ou use conhecimento externo.
    - Nunca produza opiniões ou interpretações além do que está escrito.

    EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
    Pergunta: "Qual é a capital da França?"
    Resposta: "Não tenho informações necessárias para responder sua pergunta."

    Pergunta: "Quantos clientes temos em 2024?"
    Resposta: "Não tenho informações necessárias para responder sua pergunta."

    Pergunta: "Você acha isso bom ou ruim?"
    Resposta: "Não tenho informações necessárias para responder sua pergunta."

    PERGUNTA DO USUÁRIO:
    {user_input}

    RESPONDA A "PERGUNTA DO USUÁRIO"
"""
)

llm = ChatGoogleGenerativeAI(model=os.getenv("CHAT_MODEL"), temperature=0.6)

chain = RunnableLambda(retrieve_context) | base_template | llm

if __name__ == "__main__":
    while True:
        user_question = input("Digite sua pergunta: ")
        response = chain.invoke(user_question)
        print("Resposta: ", response.content)




