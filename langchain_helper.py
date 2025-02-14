import pickle
import os
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.documents import Document

# ----------------------------
# Pickle Files
# ----------------------------
with open('pickle_files/qa_pairs.pkl', 'rb') as file:
    qa_pairs: dict[str, List[str]] = pickle.load(file)

with open('pickle_files/tags.pkl', 'rb') as file:
    tags = pickle.load(file)
print("Loaded Pickle files")



# ----------------------------
# Vector DB
# ----------------------------
vector_store = Chroma(
    collection_name='new_collection',
    embedding_function=HuggingFaceEmbeddings(),
    persist_directory='vecdb_contents'             # vector db directory
)
retriever = vector_store.as_retriever(k=3)
print("Loaded VectorStore files")



# ----------------------------
# GROQ Model
# ----------------------------
if not os.getenv('GROQ_API_KEY'):
    os.environ['GROQ_API_KEY'] = input("GROQ_API_KEY: ")

llm = ChatGroq(model_name='llama-3.3-70b-versatile')
system = """
You are a Medical Question Answering Chatbot that answers users Medical Queries, 
using this context: {context}, retrived from medical QA Dataset, 
If the users query is not related to the medical field don't answer them. 
"""

prompt_template = ChatPromptTemplate([
    ("system", system),
    ("placeholder", "{placeholder}"),
    ("human", "{user_query}")
])
chain = prompt_template | llm


def get_resp(user_query, placeholder=[]):
    questions = retriever.invoke(user_query)
    answers = ["\n".join(qa_pairs[ques.page_content]) for ques in questions]

    resp = chain.invoke({
        "context": "\n\n".join(answers),
        "placeholder": placeholder,
        "user_query": user_query
    })

    return resp.content
