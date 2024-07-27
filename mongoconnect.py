from pymongo import MongoClient

client = MongoClient(
    "mongodb+srv://sudhaneg8321:<password>@gemini.nx6gfiz.mongodb.net/"
)
dbName = "gemini_project"
collectionName = "Research_Paper"
collection = client[dbName][collectionName]
