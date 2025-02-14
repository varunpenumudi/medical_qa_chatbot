from langchain_helper import get_resp
import streamlit as st


# ----------------------------
# STREAMLIT APP
# ----------------------------
st.title(" MEDICAL QA CHATBOT 🩺")
st.write("")

# Initialize the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Messages in History first
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])


question = st.chat_input("Please Enter Your Query")

if question:
    with st.chat_message('user'):
        st.markdown(question)
    
    st.session_state.messages.append({"role":'user', 'content': question})

    placeholder = [('ai' if msg['role']=='assistant' else 'human', msg['content']) for msg in st.session_state.messages]

    resp = get_resp(question, placeholder)

    with st.chat_message('assistant'):
        st.markdown(resp)

    st.session_state.messages.append({'role': 'assistant', 'content': resp})