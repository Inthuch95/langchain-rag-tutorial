import os
import shutil

from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

class ChromaDatabase:
    def __init__(
        self, 
        chroma_path: str = "chroma", 
        data_path: str = "data", 
        chunk_size: int = 300, 
        chunk_overlap: int = 100, 
        file_type: str = "txt"
    ):
        """
        Initialize the ChromaDatabase with paths for Chroma and data.

        :param chroma_path: Path to the Chroma database.
        :param data_path: Path to the directory containing data files.
        :param chunk_size: Chunk size for splitting text.
        :param chunk_overlap: Chunk overlap for splitting text.
        :param file_type: File type to load from the data directory.
        """
        self.chroma_path = chroma_path
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.file_type = file_type
        
    def generate_data_store(self) -> None:
        """
        Generate the data store by loading documents, splitting text, and saving to Chroma.
        """
        documents = self._load_documents()
        chunks = self._split_text(documents)
        self._save_to_chroma(chunks)
        
    def load_db(self, chroma_path: str) -> None:
        """
        Load the Chroma database from the specified path.

        :param chroma_path: Path to the Chroma database.
        """
        self.db = Chroma(persist_directory=chroma_path, embedding_function=OpenAIEmbeddings())
        logger.info(f"Loaded Chroma database from {chroma_path}.")
    
    def search(self, query_text: str, k: int = 4, similarity_threshold: float = 0.7) -> list[tuple[Document, float]]:
        """
        Search the Chroma database for similar documents.
        
        :param query_text: The query text to search for.
        :param k: The number of results to return.
        :param similarity_threshold: The minimum similarity score to return results.
        :return: A list of tuples containing the Document and similarity score.
        """
        results = self.db.similarity_search_with_relevance_scores(query_text, k)
        if len(results) == 0 or results[0][1] < similarity_threshold:
            logger.info(f"Unable to find matching results.")
            raise ValueError("No matching results found.")
        return results
    
    def _load_documents(self) -> list[Document]:
        """
        Load documents from the specified data path.

        :return: A list of loaded Document objects.
        """
        logger.info(f"Loading documents from {self.data_path}.")
        loader = self._get_doc_loader()
        documents = loader.load()
        return documents
    
    def _get_doc_loader(self):
        if self.file_type == "md":
            return DirectoryLoader(self.data_path, glob=f"*.{self.file_type}")
        elif self.file_type == "pdf":
            return PyPDFDirectoryLoader(self.data_path)
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")
    
    def _split_text(self, documents: list[Document]) -> list[Document]:
        """
        Split the text of the documents into chunks.

        :param documents: A list of Document objects to be split.
        :return: A list of Document chunks.
        """
        logger.info(f"Splitting text into chunks with size {self.chunk_size} and overlap {self.chunk_overlap}.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def _save_to_chroma(self, chunks: list[Document]):
        """
        Save the chunks to the Chroma database.

        :param chunks: A list of Document chunks to be saved.
        """
        # Clear out the database first.
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)

        # Create a new DB from the documents.
        self.db = Chroma.from_documents(
            chunks, OpenAIEmbeddings(), persist_directory=self.chroma_path
        )
        self.db.persist()
        logger.info(f"Saved {len(chunks)} chunks to {self.chroma_path}.")
        