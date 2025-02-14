## Overview

This project is a Medical Question Answering (Q&A) Chatbot developed as part of an internship task. It leverages the [MedQuAD dataset](https://github.com/abachaa/MedQuAD) to provide users with answers to medical questions. 

The chatbot utilizes a **vector database for information retrieval**, powered by **ChromaDB** and **HuggingFaceEmbeddings**.  It employs **Groq's `llama-3.2-70b-model` large language model** to generate responses based on the retrieved medical information. The user interacts with the chatbot through a user-friendly chat interface built with Streamlit.

You can try this chatbot online [here](https://medicalquerychatbot.streamlit.app/)

## Project Structure

This project is organized into the following directories and files:
```
├───MedQuAD               (Dataset Folder)
├───pickle_files          (Pickle Files for QA Pairs Dictionary)
├───vecdb_contents        (Vector DataBase)
├───chatbot.py            (Streamlit Chatbot Code)
├───langchain_helper.py   (Helper Python Module for chatbot)
├───xml_to_vectordb.ipynb (Code for store the XML Dataset in Vector DB, Pickle Files)
└───README.md
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/medical_qa_chatbot
    cd medical_qa_chatbot
    ```

2.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run Streamlit file**
    ```bash
    streamlit run chatbot.py
    ```

## Features of Chatbot

* **Medical Q&A:** Answers medical questions using the MedQuAD dataset.
* **Vector DB Retrieval:** Employs ChromaDB and HuggingFace Embeddings for efficient medical information retrieval via similarity search.
* **LLM Response:** Generates natural language answers using Groq's `qwen-2.5-32b` model.
* **Streamlit Chat UI:** User-friendly web interface with Streamlit for conversational Q&A.
* **Chat History:**  Tracks conversation history within the session.
* **MedQuAD Knowledge Base:**  Leverages the MedQuAD dataset of medical question-answer pairs.
* **Optimized Data Loading:**  Uses pickled files for fast loading of question-answer data.


## Technologies Used

* **Programming Language:** Python
* **User Interface Framework:** Streamlit

* **Natural Language Processing (NLP) & Retrieval Libraries:**
  * `langchain`
  * `langchain_huggingface`
  * `langchain_groq`
  * `langchain_chroma`
  * `langchain_core`

* **Other Libraries:**
  * `pickle`
  * `os`
  * `getpass`
