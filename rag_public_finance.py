# -*- coding: utf-8 -*-
# Retrieval-Augmented Generation (RAG) Chatbot for Finance

"""Access to files"""

import os
import markdown
path = os.path.join("data", "documents")
persist_dir = os.path.join("data", "storage")


"""Large Language Model"""

import torch
from langchain_groq import ChatGroq

# Load the large language model
llm=ChatGroq(temperature=0.2, model_name="llama3-70b-8192", groq_api_key=os.environ.get('GROQ_KEY'))

"""Embedding model"""

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Load the embedding model
model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
encode_kwargs = {"normalize_embeddings": True}
embed_model = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs, cache_folder="cache"
)

"""RAG Settings"""

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Context prompt for the RAG chatbot
context_system_prompt = (
    "Given the chat history and the latest user question formulate a standalone question which can be understood without the chat history."
    "Do NOT answer the question"
    "If the question seems complex, try to decompose it in simpler steps."
    "If the question is clear and doesn't need to use the chat history, just return it as is."
)
context_prompt = ChatPromptTemplate.from_messages([
    ("system", context_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Prompt engineering
system_prompt = (
    """You are a Public Finance assistant and advisor, speciallized in tax administration, your role is to provide clear, precise and detailed but concise responses to the user questions.
If you don't know how to respond, just say it, don't try to create or imagine false information.
Answer as fluently as possible, try to keep a natural conversation.
Use markdown syntax to enhance the look and readability of your answers and jump lines to keep the conversation clear.
This are few exemples of a good comversation:
==== example 1 ====
User:What are the most important technologies to develop for tax administration by the end of 2024 ?

Assistant: **According to the provided context, the most important technologies to develop for tax administration by the end of 2024 include:**

1. **Artificial Intelligence for Automation**: Implement AI-powered systems to automate tasks, such as processing tax returns, identifying errors, and reducing manual intervention.
2. **Blockchain for Secure Data Storage**: Utilize blockchain technology to securely store tax-related data, ensuring enhanced data integrity and transparency.
3. **Cloud-Based Platforms**: Migrate tax administration systems to cloud-based platforms, enabling scalability, flexibility, and cost-effective operations.
4. **Natural Language Processing**: Integrate NLP to improve customer service, providing intuitive and personalized support to taxpayers.
5. **Predictive Analytics**: Leverage predictive analytics to identify potential tax evasion, detect anomalies, and optimize tax collection.

These technologies are considered crucial for improving tax administration efficiency, effectiveness, and responsiveness to taxpayers.
==== example 2 ====
User question:How to tackle the inflation ?

Assistant: **Addressing Inflation and Fiscal Policy**

In the context of the provided sources, addressing inflation requires a multi-faceted approach that involves fiscal policy, monetary policy, and structural reforms. According to the Fiscal Monitor (April 2024), fiscal policies should focus on directing innovation, promoting productivity growth, and ensuring the overall gains are fairly and widely shared.

To tackle inflation, the report suggests implementing fiscal policies that:

1. **Reduce fiscal deficits**: Gradually reduce fiscal deficits to stabilize debt and boost fiscal space.
2. **Promote innovation**: Encourage innovation, particularly in green sectors, to drive productivity growth and reduce emissions.
3. **Enhance tax capacity**: Strengthen institutional capacity and tax systems to mobilize revenue and reduce fiscal gaps.
4. **Implement structural reforms**: Pursue structural reforms to improve productivity, labor market flexibility, and competitiveness.

In addition, the report highlights the importance of international cooperation, institutional capacity building, and medium-term reform plans to address the challenges posed by inflation.

**Key Takeaways**

1. Fiscal policy should prioritize innovation, productivity growth, and fairness to address inflation.
2. Implementing fiscal policies that reduce deficits, promote innovation, and enhance tax capacity is crucial.
3. Structural reforms and international cooperation are essential to address the root causes of inflation."""
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker

# Text splitter for the RAG chatbot
text_splitter = SemanticChunker(
    breakpoint_threshold_type="percentile",
    embeddings=embed_model,
)

def format_docs(docs):
    # Format the documents
    return "\n\n".join(doc.page_content for doc in docs)


""" Vector Database """

# Vector database class to store the data
class VectorDatabase:
    # Intialize the vector database
    def __init__(self):
        if not os.path.exists(persist_dir):
            folder_path = os.path.join("data", "documents")
            # Load the documents
            loader = DirectoryLoader(folder_path, loader_cls=PyPDFLoader)
            documents = text_splitter.split_documents(loader.load())
            print("Documents loaded successfully")
            # Create the database
            db = FAISS.from_documents(
                documents,
                embed_model,
            )
            # Save the database to the storage
            db.save_local(persist_dir)
            print("Vector database created successfully")
        # Load the database from the storage
        self.db = FAISS.load_local(persist_dir, embed_model, allow_dangerous_deserialization=True)
        print("Vector database loaded successfully")

    # Reset the database
    def reset(self):
        # Delete the existing databse
        if os.path.exists(persist_dir):
            os.remove(persist_dir)
        # Load the documents and create the database
        self.__init__()
    

""" RAG Chatbot """

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# RAG chatbot class
class RAG:
    # Initialize the RAG chatbot
    def __init__(self, vector_database):
        self.messages = []
        self.llm = llm
        self.db = vector_database.db
        # Retrieving from FAISS database
        retriever = self.db.as_retriever(
            embed_model=embed_model,
            top_k=12,
            max_tokens=512,
            text_splitter=text_splitter,
            search_type="similarity",
        )
        # Efficient prompting to retrieve from the database
        history_aware_retreiver = create_history_aware_retriever(
            llm,
            retriever,
            context_prompt
        )
        # Prompt engineering for the RAG chatbot
        qa_chain = create_stuff_documents_chain(
            llm,
            qa_prompt,
        )
        # Create the retrieval chain
        rag_chain = create_retrieval_chain(
            history_aware_retreiver,
            qa_chain,
        )
        # Chat history for the RAG chatbot
        self.chat_history_for_chain = ChatMessageHistory()
        chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: self.chat_history_for_chain,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        self.chain = (RunnablePassthrough.assign(messages_summarizer=self.summarize_messages) | chain)

    # Summarize the past messages to reduce the chat history size
    def summarize_messages(self, chain_input):
        stored_messages = self.chat_history_for_chain.messages
        if len(stored_messages) == 0:
            return False
        summarization_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "user",
                    "Distill the above chat messages into a single summary message. Include as many specific details as you can.",
                ),
            ]
        )
        summarization_chain = summarization_prompt | self.llm
        summary_message = summarization_chain.invoke({"chat_history": stored_messages})
        self.chat_history_for_chain.clear()
        self.chat_history_for_chain.add_message(summary_message)
        return True
        
    # Respond to the user query
    def respond(self, message):
        response = markdown.markdown(self.chain.invoke(
            {"input": message},
            {"configurable": {"session_id": "unused"}},
        )["answer"])
        self.messages.append({"user": "User", "text": message})
        self.messages.append({"user": "Assistant", "text": response})

    # Reset the chat history
    def reset_history(self):
        self.chat_history_for_chain.clear()
        self.messages = []