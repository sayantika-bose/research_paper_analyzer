from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = Flask(__name__)
CORS(app)

doc_embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="retrieval_document",
    google_api_key="AIzaSyBT_cXS1-V5ggaDcx7heSHJMb0h1r-xoPU",
)

db_connection = Chroma(
    persist_directory="./chroma_db", embedding_function=doc_embeddings_model
)

retriever = db_connection.as_retriever(search_kwargs={"k": 110})

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""You are a useful AI bot, answer any question asked by the user from the specific information regarding it. Don't answer anything apart from that"""
        ),
        HumanMessagePromptTemplate.from_template(
            """Answer the following question based on the specific context. Context:{context} Question:{question} Answer:"""
        ),
    ]
)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    google_api_key="AIzaSyB3BBf69PnHSy1crohfyymSJfDmvLdRjvs",
)

output_parser = StrOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


RAG_Chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question")
    context = ""  # Retrieve context from your sources if needed
    response = RAG_Chain.invoke(question)
    return jsonify({"answer": response})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
