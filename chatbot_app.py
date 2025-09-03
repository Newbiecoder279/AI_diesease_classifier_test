import streamlit as st
import requests

# Backend API URL
API_URL = "http://127.0.0.1:8000/diagnose"

st.set_page_config(page_title="Disease Diagnosis Chatbot", page_icon="💊", layout="centered")

st.title("💊 Disease Diagnosis Chatbot")
st.write("Enter your symptoms and I will suggest possible diseases with treatments. "
         "⚠️ This is **not medical advice**, please consult a doctor for professional guidance.")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Describe your symptoms..."):
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Call FastAPI backend
    try:
        response = requests.post(API_URL, json={"symptoms": prompt, "top_k": 3})
        if response.status_code == 200:
            data = response.json()
            reply = "Here are some possible results:\n\n"
            for res in data["results"]:
                reply += f"**Disease:** {res['disease']} (confidence {res['confidence']:.2f})\n\n"
                reply += f"**Possible treatments/info:** {res['info']}\n\n---\n\n"
        else:
            reply = f"⚠️ API error {response.status_code}: {response.text}"
    except Exception as e:
        reply = f"⚠️ Could not connect to API: {e}"

    # Add assistant message
    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.chat_message("assistant").markdown(reply)
