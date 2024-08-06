from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA
from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate
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

client = MongoClient(
    "mongodb+srv://saisudhane24:Sxm9jUCXjDkXGnF9@cluster0.lgvkk8o.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)
dbName = "gemini_project"
collectionName = "Research_Paper"
collection = client[dbName][collectionName]

vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    "mongodb+srv://saisudhane24:Sxm9jUCXjDkXGnF9@cluster0.lgvkk8o.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
    dbName + "." + collectionName,
    doc_embeddings_model,
    index_name="default",
)

retriever = vector_search.as_retriever(search_kwargs={"k": 110})

prompt_template = """Interact with the user based upon their sentiment and check whether the context is related to a research paper, if it is a research paper, then answer to the questions asked by the user. If not ask the user to upload a research paper and you neednt answer the question asked by the user.

{context}

Question: {question}
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    google_api_key="AIzaSyB3BBf69PnHSy1crohfyymSJfDmvLdRjvs",
)

output_parser = StrOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question")
    RAG_Chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | chat_model
        | output_parser
    )

    response = RAG_Chain.invoke(question)
    return jsonify({"answer": response})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
