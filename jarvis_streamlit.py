import os
import json
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import psutil

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("❌ OPENROUTER_API_KEY not found!")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

st.set_page_config(
    page_title="JARVIS - AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        .stApp { background-color: #0a0e27; }
        h1, h2, h3 { color: #4fc3f7; }
        .stChatMessage { background-color: #1a1f3a; border-left: 3px solid #4fc3f7; }
    </style>
""", unsafe_allow_html=True)

MEMORY_FILE = "jarvis_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(chat_history):
    with open(MEMORY_FILE, "w") as f:
        json.dump(chat_history, f, indent=2)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_memory()

st.markdown("<h1 style='text-align:center'>🤖 JARVIS</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#4fc3f7'>▓▓▓ MEMORY ONLINE ▓▓▓</p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### SYSTEM STATUS")
    st.write(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
    st.write(f"💾 RAM: {psutil.virtual_memory().percent}%")
    st.write(f"⚙️ CPU: {psutil.cpu_percent()}%")
    st.divider()
    total = len(st.session_state.chat_history)
    st.write(f"📊 Total Messages: {total}")
    if st.button("🗑️ Clear Memory"):
        st.session_state.chat_history = []
        save_memory([])
        st.rerun()

for msg in st.session_state.chat_history[-20:]:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        role = "👤 SIR" if msg["role"] == "user" else "🤖 JARVIS"
        st.markdown(f"**{role}:** {msg['content']}")

user_input = st.chat_input("Speak or type, Sir...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    save_memory(st.session_state.chat_history)
    with st.chat_message("user"):
        st.markdown(f"**👤 SIR:** {user_input}")
    with st.chat_message("assistant"):
        with st.spinner("🔄 Recalling..."):
            try:
                messages = [{"role": "system", "content": "You are JARVIS. Be witty and British. Call the user Sir."}]
                messages.extend(st.session_state.chat_history[-10:])
                completion = client.chat.completions.create(
                    extra_headers={"HTTP-Referer": "http://localhost", "X-Title": "Jarvis"},
                    model="openrouter/auto",
                    messages=messages
                )
                response = completion.choices[0].message.content
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                save_memory(st.session_state.chat_history)
                st.markdown(f"**🤖 JARVIS:** {response}")
            except Exception as e:
                st.error(f"SYSTEM ERROR: {str(e)}")
    st.rerun()
