import streamlit as st

st.set_page_config(page_title="Login | ZenBot", page_icon="🔐")

st.title("🔐 Login to ZenBot")

# Initialize session state for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Dummy user credentials
USERNAME = "admin"
PASSWORD = "password123"

# Login form
username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    if username == USERNAME and password == PASSWORD:
        st.session_state.authenticated = True
        st.success("✅ Login successful! Redirecting...")
        st.rerun()
    else:
        st.error("❌ Invalid username or password")

# Redirect if logged in
if st.session_state.authenticated:
    st.switch_page("pages/chatbot.py")
