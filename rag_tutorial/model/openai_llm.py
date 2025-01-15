from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from loguru import logger

class OpenAIModel():
    prompt_template = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """
    
    def __init__(self, name: str):
        """
        Initialize the OpenAIModel with the model name.
        
        :param name: The name of the model to use.
        """
        self.model = ChatOpenAI(name=name)
        
    def predict(self, query_text: str, documents: list[tuple[Document, float]]) -> str:
        """
        Predict the response to the query text based on the documents.
        
        :param query_text: The query text to predict a response for.
        :param documents: A list of tuples containing the Document and similarity score.
        :return: The response text from llm.
        """
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in documents])
        prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
        prompt = prompt_template.format(context=context_text, question=query_text)
        logger.info(prompt)
        return self.model.predict(prompt)
    