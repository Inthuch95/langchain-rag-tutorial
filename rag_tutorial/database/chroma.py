import os
import shutil

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

class ChromaDatabase:
    
    def __init__(self, chroma_path: str, data_path: str, chunk_size: int = 300, chunk_overlap: int = 100):
        """
        Initialize the ChromaDatabase with paths for Chroma and data.

        :param chroma_path: Path to the Chroma database.
        :param data_path: Path to the directory containing data files.
        :param chunk_size: Chunk size for splitting text.
        :param chunk_overlap: Chunk overlap for splitting text.
        """
        self.chroma_path = chroma_path
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def generate_data_store(self) -> None:
        """
        Generate the data store by loading documents, splitting text, and saving to Chroma.
        """
        documents = self._load_documents()
        chunks = self._split_text(documents)
        self._save_to_chroma(chunks)
        
    def _load_documents(self) -> list[Document]:
        """
        Load documents from the specified data path.

        :return: A list of loaded Document objects.
        """
        loader = DirectoryLoader(self.data_path, glob="*.md")
        documents = loader.load()
        return documents
    
    def _split_text(self, documents: list[Document]) -> list[Document]:
        """
        Split the text of the documents into chunks.

        :param documents: A list of Document objects to be split.
        :return: A list of Document chunks.
        """
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
        db = Chroma.from_documents(
            chunks, OpenAIEmbeddings(), persist_directory=self.chroma_path
        )
        db.persist()
        logger.info(f"Saved {len(chunks)} chunks to {self.chroma_path}.")
        