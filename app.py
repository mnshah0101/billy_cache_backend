from flask import Flask, request, jsonify
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import time
from datetime import datetime
import dotenv
import os
from flask_cors import CORS


dotenv.load_dotenv()

app = Flask(__name__)
CORS(app)


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Define your Pinecone index name
INDEX_NAME = os.getenv('PINECONE_INDEX')

# Create or connect to the Pinecone index
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(INDEX_NAME)

# Function to get embedding


def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# Function to embed and upload question


def embed_and_upload_question(question, sql_query, model="text-embedding-3-large"):
    embedding = get_embedding(question, model)
    question_id = str(hash(question + str(time.time())))  # Unique ID
    metadata = {
        "question": question,
        "sql_query": sql_query,
        "date_added": datetime.utcnow().isoformat()
    }
    index.upsert(vectors=[(question_id, embedding, metadata)])
    return question_id

# Route to upload a new embedding


@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    question = data.get('question')
    sql_query = data.get('sql_query')
    if not question or not sql_query:
        return jsonify({"error": "Both 'question' and 'sql_query' are required."}), 400
    question_id = embed_and_upload_question(question, sql_query)
    return jsonify({"message": "Uploaded successfully", "question_id": question_id}), 200

# Route to search for similar questions


@app.route('/search', methods=['GET'])
def search():
    question = request.args.get('question')
    k = int(request.args.get('k', 5))  # Default k=5
    if not question:
        return jsonify({"error": "'question' parameter is required."}), 400
    embedding = get_embedding(question)
    # Query Pinecone index
    search_response = index.query(
        vector=embedding, top_k=k, include_metadata=True)
    results = []
    for match in search_response.matches:
        metadata = match.metadata
        results.append({
            "score": match.score,
            "question": metadata.get('question'),
            "sql_query": metadata.get('sql_query'),
            "date_added": metadata.get('date_added'),
            "index_id": match.id
        })
    return jsonify({"results": results}), 200

# Route to delete an embedding by question_id


@app.route('/delete/<question_id>', methods=['DELETE'])
def delete(question_id):
    index.delete(ids=[question_id])
    return jsonify({"message": f"Question {question_id} deleted successfully."}), 200


if __name__ == '__main__':
    app.run(debug=True)
