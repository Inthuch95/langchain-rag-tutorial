import argparse
import openai 
import os

from dotenv import load_dotenv
from loguru import logger
from rag_tutorial.model.openai_llm import OpenAIModel
from rag_tutorial.utils.config import read_config
from rag_tutorial.utils.database import init_vector_database


load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
CONFIG_PATH = "configs/configs.yaml"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    
    config = read_config(CONFIG_PATH)
    db_config = config["database"]
    app_config = config["app"]
    
    # Load LLM model and vector database
    model = OpenAIModel(app_config["model_name"])
    vector_db = init_vector_database(app_config, db_config)
    documents = vector_db.search(query_text, k=app_config["k"], similarity_threshold=app_config["similarity_threshold"])
    
    # Send query to LLM and get response
    response_text = model.predict(query_text, documents)
    logger.info(f"Response: {response_text}")
    
    
if __name__ == "__main__":
    main()
    