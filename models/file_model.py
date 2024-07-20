# models/file_model.py
def save_file_details(mongo, filename, file_path):
    # Create a record to insert into MongoDB
    file_record = {"filename": filename, "path": file_path}

    # Insert the record into the 'files' collection
    mongo.db.files.insert_one(file_record)

    return file_record
