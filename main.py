import argparse
import openai 
import os

from dotenv import load_dotenv
from loguru import logger
from rag_tutorial.database.chroma import ChromaDatabase
from rag_tutorial.model.openai_llm import OpenAIModel
from rag_tutorial.utils.config import read_config


load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
CONFIG_PATH = "configs/configs.yaml"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    
    config = read_config(CONFIG_PATH)
    db_config = config["database"]
    app_config = config["app"]
    
    # Initialize vector DB
    if app_config["create_db"]:
        vector_db = ChromaDatabase(
            db_config["chroma_path"], 
            db_config["data_path"],
            db_config["chunk_size"],
            db_config["chunk_overlap"],
            db_config["file_type"]
        )
        vector_db.generate_data_store()
    else:
        vector_db = ChromaDatabase()
        vector_db.load_db(db_config["chroma_path"])
    
    # Load LLM and send query.
    model = OpenAIModel(app_config["model_name"])
    documents = vector_db.search(query_text, k=app_config["k"], similarity_threshold=app_config["similarity_threshold"])
    response_text = model.predict(query_text, documents)
    logger.info(f"Response: {response_text}")
    
    
if __name__ == "__main__":
    main()
    