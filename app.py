from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity,
)
from werkzeug.security import check_password_hash
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
from flask_session import Session
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"  # Define your upload folder
app.config["MONGO_URI"] = os.getenv("MONGO_URI")  # MongoDB URI
mongo = MongoClient(app.config["MONGO_URI"])
CORS(app)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")
jwt = JWTManager(app)
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

# Use environment variables for database and collection names
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class User(UserMixin):
    def _init_(self, email, password=None, name=None, phone=None):
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
        access_token = create_access_token(
            identity={"email": user.email, "name": user.name}
        )
        return (
            jsonify({"message": "Login successful", "access_token": access_token}),
            200,
        )
    return jsonify({"message": "Invalid email or password"}), 401


@app.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    # Access the identity of the current user with get_jwt_identity
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user), 200


@app.route("/logout", methods=["POST"])
def logout():
    collection = mongo[DB_NAME][COLLECTION_NAME]
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
        model=MODEL_NAME,
        task_type="retrieval_document",
        google_api_key=GOOGLE_API_KEY,
    )
    collection = mongo[DB_NAME][COLLECTION_NAME]

    MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=doc_embeddings_model,
        collection=collection,
        index_name="default",
    )


if __name__ == "__main__":
    app.run(debug=True)