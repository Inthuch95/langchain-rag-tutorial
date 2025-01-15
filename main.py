import openai 
import os

from dotenv import load_dotenv
from rag_tutorial.database.chroma import ChromaDatabase
from rag_tutorial.utils.config import read_config

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
CONFIG_PATH = "configs/configs.yaml"

if __name__ == "__main__":
    config = read_config(CONFIG_PATH)
    db_config = config["database"]
    chroma_db = ChromaDatabase(
        db_config["chroma_path"], 
        db_config["data_path"],
        db_config["chunk_size"],
        db_config["chunk_overlap"]
    )
    chroma_db.generate_data_store()
    