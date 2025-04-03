import streamlit as st

# Set Streamlit page config
st.set_page_config(page_title="ZenBot", page_icon="🤖", layout="centered")

st.title("Welcome to ZenBot! 🤖")
st.write("""
ZenBot is an AI-powered chatbot that retrieves information from documents and provides accurate answers using Mistral-7B.

### Features:
✅ Retrieve information from PDFs  
✅ AI-powered responses using LangChain + FAISS  
✅ Chat history saved for better experience  

👉 **[Login Here](./pages/login.py)** to start chatting!
""")

# Optional: Add a navigation button
if st.button("Go to Login"):
    st.switch_page("pages/login.py")
