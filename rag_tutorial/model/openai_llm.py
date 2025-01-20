from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

class OpenAIModel():
    prompt_template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer as concise as possible:

    {context}

    ---

    Question: {question}
    """
    
    def __init__(self, name: str):
        """
        Initialize the OpenAIModel with the model name.
        
        :param name: The name of the model to use.
        """
        self.model = ChatOpenAI(name=name)
        self.parser = StrOutputParser()
        
    def predict(self, query_text: str, documents: list[tuple[Document, float]]) -> str:
        """
        Predict the response to the query text based on the documents.
        
        :param query_text: The query text to predict a response for.
        :param documents: A list of tuples containing the Document and similarity score.
        :return: The response text from llm.
        """
        prompt = self._format_prompt(query_text, documents)
        logger.info(prompt)
        chain = self.model | self.parser
        return chain.invoke(prompt)
    
    def _format_prompt(self, query_text: str, documents: list[tuple[Document, float]]) -> str:
        """
        Format the prompt with the query text and documents.
        
        :param query_text: The query text to format.
        :param documents: A list of tuples containing the Document and similarity score.
        :return: The formatted prompt.
        """
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in documents])
        prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
        prompt = prompt_template.format(context=context_text, question=query_text)
        return prompt
    