from rag_tutorial.database.chroma import ChromaDatabase

def init_vector_database(app_config: dict, db_config: dict) -> ChromaDatabase:
    """
    Initialize the ChromaDatabase based on the application and database configurations.

    :param app_config: The application configuration.
    :param db_config: The database configuration.
    :return: An instance of ChromaDatabase.
    """
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
    return vector_db
