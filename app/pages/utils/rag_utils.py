import streamlit as st
from datetime import datetime
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm
import json
import os
from typing import List, Dict
from pymongo import MongoClient

from fireworks.client import Fireworks

fw_client = Fireworks(
    api_key=st.secrets["fireworks"]["api_key"]
)
model = "accounts/fireworks/models/llama-v3-8b-instruct"

# Name of the database -- Change if needed or leave as is
DB_NAME = "mongodb_rag_lab"
# Name of the collection -- Change if needed or leave as is
COLLECTION_NAME = "knowledge_base"
# Name of the vector search index -- Change if needed or leave as is
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"


mongodb_client = MongoClient(st.secrets["mongo"]["host"], appname="devrel.workshop.rag")
collection = mongodb_client[DB_NAME][COLLECTION_NAME]

history_collection = mongodb_client[DB_NAME]["chat_history"]

def store_chat_message(session_id: str, role: str, content: str) -> None:
    """
    Store a chat message in a MongoDB collection.

    Args:
        session_id (str): Session ID of the message.
        role (str): Role for the message. One of `system`, `user` or `assistant`.
        content (str): Content of the message.
    """
    # Create a message object with `session_id`, `role`, `content` and `timestamp` fields
    # `timestamp` should be set the current timestamp
    message = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": datetime.now(),
    }
    # Insert the `message` into the `history_collection` collection
    history_collection.insert_one(message)

embedding_model = SentenceTransformer("thenlper/gte-small")

def get_embedding(text: str) -> List[float]:
    """
    Generate the embedding for a piece of text.

    Args:
        text (str): Text to embed.

    Returns:
        List[float]: Embedding of the text as a list.
    """
    embedding = embedding_model.encode(text)

    return embedding.tolist()


def vector_search(user_query: str) -> List[Dict]:
    """
    Retrieve relevant documents for a user query using vector search.

    Args:
    user_query (str): The user's query string.

    Returns:
    list: A list of matching documents.
    """

    # Generate embedding for the `user_query` using the `get_embedding` function defined in Step 5
    query_embedding = get_embedding(user_query)

    # Define an aggregation pipeline consisting of a $vectorSearch stage, followed by a $project stage
    # Set the number of candidates to 150 and only return the top 5 documents from the vector search
    # In the $project stage, exclude the `_id` field and include only the `body` field and `vectorSearchScore`
    # NOTE: Use variables defined previously for the `index`, `queryVector` and `path` fields in the $vectorSearch stage
    pipeline = [
      {
          "$vectorSearch": {
              "index": ATLAS_VECTOR_SEARCH_INDEX_NAME,
              "queryVector": query_embedding,
              "path": "embedding",
              "numCandidates": 150,
              "limit": 5,
          }
      },
      {
          "$project": {
              "_id": 0,
              "Scraped Content": 1,
              "score": {"$meta": "vectorSearchScore"}
          }
      }
  ]

    # Execute the aggregation `pipeline` and store the results in `results`
    results = collection.aggregate(pipeline)

    return list(results)



def create_prompt(user_query: str) -> str:
    """
    Create a chat prompt that includes the user query and retrieved context.

    Args:
        user_query (str): The user's query string.

    Returns:
        str: The chat prompt string.
    """
    # Retrieve the most relevant documents for the `user_query` using the `vector_search` function defined in Step 8
    context = vector_search(user_query)
    # Extract the "body" field from each document in `context`
    documents = "\n\n".join([d.get("Scraped Content", "") for d in context])

    # Prompt consisting of the question and relevant context to answer it
    prompt = f"Answer the question based only on the following context. If the context is empty, say I DON'T KNOW\n\nContext:\n{documents}\n\nQuestion:{user_query}"
    return prompt




def retrieve_session_history(session_id: str) -> List:
    """
    Retrieve chat message history for a particular session.

    Args:
        session_id (str): Session ID to retrieve chat message history for.

    Returns:
        List: List of chat messages.
    """
    # Query the `history_collection` collection for documents where the "session_id" field has the value of the input `session_id`
    # Sort the results in increasing order of the values in `timestamp` field
    cursor =  history_collection.find({"session_id": session_id}).sort("timestamp", 1)

    if cursor:
        # Iterate through the cursor and extract the `role` and `content` field from each entry
        # Then format each entry as: {"role": <role_value>, "content": <content_value>}
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in cursor]
    else:
        # If cursor is empty, return an empty list
        messages = []

    return messages




def generate_answer(session_id: str, user_query: str) -> None:
    """
    Generate an answer to the user's query taking chat history into account.

    Args:
        session_id (str): Session ID to retrieve chat history for.
        user_query (str): The user's query string.
    """
    messages = []

    # Retrieve documents relevant to the user query and convert them to a single string
    context = vector_search(user_query)
    context = "\n\n".join([d.get("Scraped Content", "") for d in context])
    # Create a system prompt containing the retrieved context
    system_message = {
        "role": "system",
        "content": f"Answer the question based only on the following context. If the context is empty, say I DON'T KNOW\n\nContext:\n{context}",
    }
    # Append the system prompt to the `messages` list
    messages.append(system_message)

    # Use the `retrieve_session_history` function to retrieve message history from MongoDB for the session ID `session_id`
    # And add all messages in the message history to the `messages` list
    message_history = retrieve_session_history(session_id)

    messages.extend(message_history)

    # Format the user message in the format {"role": <role_value>, "content": <content_value>}
    # The role value for user messages must be "user"
    # And append the user message to the `messages` list
    user_message = {"role": "user", "content": user_query}


    messages.append(user_message)

    # Call the chat completions API
    response = fw_client.chat.completions.create(model=model, messages=messages)

    # Extract the answer from the API response
    answer = response.choices[0].message.content

    # Use the `store_chat_message` function to store the user message and also the generated answer in the message history collection
    # The role value for user messages is "user", and "assistant" for the generated answer
    store_chat_message(session_id, "user", user_query)
    store_chat_message(session_id, "assistant", answer)


    return answer

