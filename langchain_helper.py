import pickle
import os
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
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
You are a highly knowledgeable and reliable Medical Question Answering AI, 
designed to provide accurate, evidence-based responses using {context} from the Medical QA Dataset. 
Ensure answers are precise, relevant, and easy to understand while maintaining scientific accuracy. 
If the dataset lacks sufficient information, acknowledge this and recommend consulting a healthcare professional. 
Avoid medical diagnosis or treatment advice, emphasizing that responses do not replace professional care. 
Use clear, empathetic language, especially for sensitive topics, and structure responses with a brief summary, 
detailed explanation, and next steps if applicable. If uncertain, guide users to appropriate medical resources.
"""

prompt_template = ChatPromptTemplate([
    ("system", system),
    ("placeholder", "{placeholder}"),
    ("human", "{user_query}")
])
chain = prompt_template | llm


def get_resp(user_query, placeholder=[]):
    questions = retriever.invoke(user_query)
    answers = [ques.page_content + "\n" + "\n".join(qa_pairs[ques.page_content]) for ques in questions]

    resp = chain.invoke({
        "context": "\n\n".join(answers),
        "placeholder": placeholder,
        "user_query": user_query
    })

    return resp.content
