import argparse
import openai 
import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from loguru import logger
from rag_tutorial.database.chroma import ChromaDatabase
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
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    
    config = read_config(CONFIG_PATH)
    db_config = config["database"]
    app_config = config["app"]
    
    # Vector DB
    if app_config["create_db"]:
        chroma_db = ChromaDatabase(
            db_config["chroma_path"], 
            db_config["data_path"],
            db_config["chunk_size"],
            db_config["chunk_overlap"],
            db_config["file_type"]
        )
        chroma_db.generate_data_store()
    
    embedding_function = OpenAIEmbeddings()
    vector_db = Chroma(persist_directory=app_config["chroma_path"], embedding_function=embedding_function)
    
     # Search the DB.
    results = vector_db.similarity_search_with_relevance_scores(query_text, k=app_config["k"])
    if len(results) == 0 or results[0][1] < app_config["similarity_threshold"]:
        logger.info(f"Unable to find matching results.")
        return
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    logger.info(prompt)
    
    model = ChatOpenAI(name=app_config["model"])
    response_text = model.predict(prompt)
    
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    formatted_response = f"Response: {response_text}"
    logger.info(formatted_response)
    
    
if __name__ == "__main__":
    main()
    