from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
from flask_login import (
    LoginManager,
    login_required,
    login_user,
    logout_user,
    current_user,
    UserMixin,
)
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import NLTKTextSplitter
from langchain_chroma import Chroma
from flask_session import Session
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"  # Define your upload folder
app.config["MONGO_URI"] = "mongodb://127.0.0.1:27017/gemini"  # MongoDB URI
mongo = PyMongo(app)
CORS(app)

app.config["SECRET_KEY"] = (
    "9f0cb7d7116f1092ef1fc972315480321215f7f515ef2cc6852663c65c2c2540"
)
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "uploads")
Session(app)

# Flask-Login Configuration
login_manager = LoginManager(app)
login_manager.login_view = "login"
bcrypt = Bcrypt(app)

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


class User(UserMixin):
    def __init__(self, email, password=None, name=None, phone=None):
        self.email = email
        self.password = password
        self.name = name
        self.phone = phone

    def get_id(self):
        return self.email

    @staticmethod
    def get(email):
        user_data = mongo.db.users.find_one({"email": email})
        if user_data:
            return User(
                email=user_data["email"],
                password=user_data["password"],
                name=user_data.get("name"),
                phone=user_data.get("phone"),
            )
        return None

    @staticmethod
    def create(email, password, name, phone):
        hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
        user_data = {
            "email": email,
            "password": hashed_password,
            "name": name,
            "phone": phone,
        }
        mongo.db.users.insert_one(user_data)


@login_manager.user_loader
def load_user(email):
    return User.get(email)


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    name = data.get("name")
    phone = data.get("phone")
    if User.get(email):
        return jsonify({"message": "Email address already exists"}), 400
    User.create(email, password, name, phone)
    return jsonify({"message": "Registration successful"}), 201


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    user = User.get(email)
    if user and bcrypt.check_password_hash(user.password, password):
        login_user(user)
        return (
            jsonify(
                {
                    "message": "Login successful",
                    "user": {
                        "email": user.email,
                        "name": user.name,
                        "phone": user.phone,
                    },
                }
            ),
            200,
        )
    return jsonify({"message": "Invalid email or password"}), 401


@app.route("/logout", methods=["POST"])
def logout():
    client = MongoClient(
        "mongodb+srv://sudhaneg8321:H3ltadIghy1M1xUK@gemini.nx6gfiz.mongodb.net/"
    )
    dbName = "gemini_project"
    collectionName = "Research_Paper"
    collection = client[dbName][collectionName]
    collection.delete_many({})

    # Clear session
    logout_user()
    return jsonify({"message": "Logout successful"}), 200


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Save file path to MongoDB
        file_data = {"filename": filename, "filepath": file_path}
        mongo.db.files.insert_one(file_data)

        process_file(file_path)

        return (
            jsonify({"message": "File uploaded successfully", "file_path": file_path}),
            200,
        )


def process_file(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = NLTKTextSplitter(chunk_size=2000, chunk_overlap=150)
    chunks = text_splitter.split_documents(pages)
    print(len(chunks))

    doc_embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="retrieval_document",
        google_api_key="AIzaSyBT_cXS1-V5ggaDcx7heSHJMb0h1r-xoPU",
    )
    client = MongoClient(
        "mongodb+srv://sudhaneg8321:H3ltadIghy1M1xUK@gemini.nx6gfiz.mongodb.net/"
    )
    dbName = "gemini_project"
    collectionName = "Research_Paper"
    collection = client[dbName][collectionName]

    MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=doc_embeddings_model,
        collection=collection,
        index_name="default",
    )


if __name__ == "__main__":
    app.run(debug=True)
