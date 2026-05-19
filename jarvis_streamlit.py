import os
import json
import base64
import requests
import pytz
import time
import math
import re
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from groq import Groq
import psutil

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY", "")
supabase_url = os.getenv("SUPABASE_URL", "")
supabase_key = os.getenv("SUPABASE_KEY", "")

if not api_key:
    st.error("❌ GROQ_API_KEY not found!")
    st.stop()

client = Groq(api_key=api_key)
IST = pytz.timezone('Asia/Kolkata')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HELIX",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ──────────────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

D = {
    "bg":        "#0d0f1a" if st.session_state.dark_mode else "#f5f5f0",
    "surface":   "#161926" if st.session_state.dark_mode else "#ffffff",
    "surface2":  "#1e2235" if st.session_state.dark_mode else "#f0f0ea",
    "border":    "#2a2f4a" if st.session_state.dark_mode else "#e0e0d8",
    "text":      "#e8eaf6" if st.session_state.dark_mode else "#1a1a2e",
    "muted":     "#6b7280" if st.session_state.dark_mode else "#9ca3af",
    "accent":    "#7c6af7",          # purple – same in both modes
    "accent2":   "#06b6d4",          # cyan accent
    "user_bg":   "#1e2235" if st.session_state.dark_mode else "#ede9fe",
    "ai_bg":     "#161926" if st.session_state.dark_mode else "#ffffff",
    "code_bg":   "#0a0c14" if st.session_state.dark_mode else "#1e1e2e",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

/* ── Reset & base ── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; }}
html, body, .stApp {{ background-color: {D['bg']} !important; color: {D['text']}; font-family: 'DM Sans', sans-serif; }}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {{ display: none !important; }}
.stDeployButton {{ display: none !important; }}
[data-testid="stToolbar"] {{ display: none !important; }}

/* ── Layout ── */
.main .block-container {{
    max-width: 820px !important;
    margin: 0 auto !important;
    padding: 0 20px 120px !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {D['surface']} !important;
    border-right: 1px solid {D['border']} !important;
}}
[data-testid="stSidebar"] * {{ color: {D['text']} !important; }}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {{
    background: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
    box-shadow: none !important;
}}
[data-testid="stChatMessage"] > div {{
    background: transparent !important;
}}

/* ── Input box ── */
[data-testid="stChatInput"] textarea {{
    background: {D['surface2']} !important;
    color: {D['text']} !important;
    border: 1.5px solid {D['border']} !important;
    border-radius: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    padding: 14px 16px !important;
    transition: border-color 0.2s !important;
    resize: none !important;
}}
[data-testid="stChatInput"] textarea:focus {{
    border-color: {D['accent']} !important;
    box-shadow: 0 0 0 3px {D['accent']}22 !important;
    outline: none !important;
}}
[data-testid="stChatInput"] button {{
    background: {D['accent']} !important;
    border-radius: 10px !important;
    border: none !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background: transparent !important;
    border: 1px solid {D['border']} !important;
    color: {D['text']} !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    transition: all 0.15s !important;
}}
.stButton > button:hover {{
    border-color: {D['accent']} !important;
    color: {D['accent']} !important;
    background: {D['accent']}11 !important;
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background: {D['surface2']} !important;
    border: 1.5px dashed {D['border']} !important;
    border-radius: 12px !important;
    padding: 8px !important;
}}
[data-testid="stFileUploader"] * {{ color: {D['muted']} !important; font-size: 13px !important; }}

/* ── Divider ── */
hr {{ border-color: {D['border']} !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {D['border']}; border-radius: 2px; }}

/* ── Animations ── */
@keyframes fadeSlideIn {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0.4; }}
}}
@keyframes spin {{
    to {{ transform: rotate(360deg); }}
}}
</style>
""", unsafe_allow_html=True)

# ── Supabase helpers ───────────────────────────────────────────────────────────
def _sb_headers():
    return {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
    }

def supabase_save(user_id, role, content):
    if not supabase_url or not supabase_key:
        return
    try:
        requests.post(
            f"{supabase_url}/rest/v1/conversations",
            headers=_sb_headers(),
            json={"user_id": user_id, "role": role, "content": content},
            timeout=5,
        )
    except Exception as e:
        print(f"[Supabase save error] {e}")

def supabase_load(user_id):
    if not supabase_url or not supabase_key:
        return []
    try:
        r = requests.get(
            f"{supabase_url}/rest/v1/conversations?user_id=eq.{user_id}&order=id.asc",
            headers=_sb_headers(),
            timeout=5,
        )
        if r.status_code == 200:
            return [{"role": m["role"], "content": m["content"]} for m in r.json()]
    except Exception as e:
        print(f"[Supabase load error] {e}")
    return []

def supabase_clear(user_id):
    if not supabase_url or not supabase_key:
        return
    try:
        requests.delete(
            f"{supabase_url}/rest/v1/conversations?user_id=eq.{user_id}",
            headers=_sb_headers(),
            timeout=5,
        )
    except Exception as e:
        print(f"[Supabase clear error] {e}")

# ── Local memory fallback ──────────────────────────────────────────────────────
MEMORY_FILE = "helix_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            return json.load(f)
    return []

def save_memory(history):
    with open(MEMORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# ── Tool functions ─────────────────────────────────────────────────────────────
def get_weather(location="London"):
    try:
        r = requests.get(f"https://wttr.in/{location}?format=j1", timeout=5)
        if r.status_code == 200:
            c = r.json()["current_condition"][0]
            return {
                "location": location,
                "temperature": c["temp_C"],
                "description": c["weatherDesc"][0]["value"],
                "humidity": c["humidity"],
                "wind_speed": c["windspeedKmph"],
                "feels_like": c["FeelsLikeC"],
            }
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Could not fetch weather"}

def get_news(query="latest", country="us"):
    if not news_api_key:
        return {"error": "NEWS_API_KEY not configured"}
    try:
        if query == "latest":
            url = f"https://newsapi.org/v2/top-headlines?country={country}&apiKey={news_api_key}"
        else:
            url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={news_api_key}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            arts = r.json().get("articles", [])[:5]
            return {"articles": [{"title": a["title"], "source": a["source"]["name"],
                                   "description": a["description"], "url": a["url"]} for a in arts]}
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Could not fetch news"}

def web_search(query):
    try:
        wiki = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}",
            timeout=5,
        )
        if wiki.status_code == 200:
            d = wiki.json()
            if d.get("extract"):
                return {"results": [{"title": d.get("title", query),
                                      "snippet": d.get("extract", "")[:500],
                                      "url": d.get("content_urls", {}).get("desktop", {}).get("page", "")}]}
        r = requests.get("https://api.duckduckgo.com/", params={"q": query, "format": "json", "no_redirect": 1}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            results = []
            if data.get("AbstractText"):
                results.append({"title": data.get("Heading", "Result"),
                                 "url": data.get("AbstractURL", ""),
                                 "snippet": data.get("AbstractText", "")})
            for t in data.get("RelatedTopics", [])[:4]:
                if "Text" in t:
                    results.append({"title": t.get("Text", "")[:80],
                                    "url": t.get("FirstURL", ""),
                                    "snippet": t.get("Text", "")})
            return {"results": results[:5]} if results else {"error": "No results"}
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Could not search"}

def calculate(expression):
    try:
        expr = expression.lower().strip()
        for old, new in [("×","*"),("÷","/"),("^","**"),("pi",str(math.pi)),
                         ("square root of","math.sqrt"),("sqrt of","math.sqrt"),("sqrt","math.sqrt"),
                         ("sin","math.sin"),("cos","math.cos"),("tan","math.tan"),("log","math.log10")]:
            expr = expr.replace(old, new)
        expr = re.sub(r'math\.sqrt\s+(\d+)', r'math.sqrt(\1)', expr)
        check = re.sub(r'[0-9\s\+\-\*\/\.\(\)e]', '',
                       expr.replace("math.sqrt","").replace("math.sin","").replace("math.cos","")
                           .replace("math.tan","").replace("math.log10","").replace("math.pi",""))
        if check == "":
            result = eval(expr, {"__builtins__": {}}, {"math": math})
            return {"result": round(float(result), 6)}
        return {"error": "Invalid expression"}
    except Exception as e:
        return {"error": str(e)}

# ── Intent detection ───────────────────────────────────────────────────────────
def detect_intent(text):
    low = text.lower()
    math_pat = re.search(r'[\d]+[\s]*[\+\-\*\/\^\×\÷][\s]*[\d]+', text)
    sqrt_pat = re.search(r'(square root of|sqrt\s+of|sqrt)\s*[\d]+', low)
    if math_pat or sqrt_pat or any(w in low for w in ["calculate","compute"]):
        if sqrt_pat:
            n = re.search(r'[\d]+', sqrt_pat.group())
            if n:
                return {"type": "calculator", "expression": f"math.sqrt({n.group()})"}
        if math_pat:
            return {"type": "calculator", "expression": math_pat.group().strip()}
        e = re.search(r'[\d\s\+\-\*\/\.\(\)\^]+', text)
        if e:
            return {"type": "calculator", "expression": e.group().strip()}

    events = ["who won","winner","champion","scored","elected","released","launched","died","born",
              "married","arrested","2024","2025","2026","latest","recent","current","today","yesterday",
              "this year","this month","ipl","cricket","match","tournament","election","movie","song"]
    if any(w in low for w in events):
        return {"type": "search", "query": text}

    if any(w in low for w in ["weather","temperature","forecast","climate","rain","snow"]):
        location = "London"
        if "in " in low:
            parts = low.split("in ")
            if len(parts) > 1:
                location = parts[1].split()[0].capitalize()
        return {"type": "weather", "location": location}

    if any(w in low for w in ["news","headlines","breaking"]):
        query = "latest"
        for w in ["about","regarding","on","for"]:
            if w in low:
                parts = low.split(w)
                if len(parts) > 1:
                    query = parts[1].strip()
                    break
        return {"type": "news", "query": query}

    if any(w in low for w in ["search","find","look up","google","web","lookup"]):
        q = text
        for p in ["search for","search","find","look up","lookup"]:
            if p in low:
                q = text.split(p, 1)[1].strip()
                break
        return {"type": "search", "query": q}

    return None

def needs_web_search(text):
    phrases = ["i don't know","i'm not sure","i cannot find","i don't have information",
               "beyond my knowledge","i'm unable to","not in my knowledge","i lack information",
               "i do not have","cannot recall","not aware of","up-to-date","most recent",
               "latest information","don't have access","cannot access","no information"]
    return any(p in text.lower() for p in phrases)

# ── Artifact renderer ──────────────────────────────────────────────────────────
def render_message(content: str):
    """
    Parse message and render code blocks as styled artifacts,
    everything else as markdown.
    """
    code_pattern = re.compile(r'```(\w*)\n?(.*?)```', re.DOTALL)
    parts = []
    last = 0
    for m in code_pattern.finditer(content):
        if m.start() > last:
            parts.append(("text", content[last:m.start()]))
        parts.append(("code", m.group(1) or "text", m.group(2).rstrip()))
        last = m.end()
    if last < len(content):
        parts.append(("text", content[last:]))

    for part in parts:
        if part[0] == "text":
            st.markdown(part[1])
        else:
            lang = part[1]
            code = part[2]
            # Styled artifact block
            st.markdown(f"""
<div style="
    background:{D['code_bg']};
    border:1px solid {D['border']};
    border-radius:12px;
    overflow:hidden;
    margin:10px 0;
    font-family:'JetBrains Mono',monospace;
    font-size:13px;
    animation: fadeSlideIn 0.3s ease;
">
  <div style="
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:8px 14px;
    background:{D['surface2']};
    border-bottom:1px solid {D['border']};
  ">
    <span style="color:{D['accent']};font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">
      {'🔵' if lang in ('python','py') else '🟡' if lang in ('js','javascript','ts','typescript') else '🔴' if lang in ('html','css') else '⚪'} {lang or 'code'}
    </span>
    <span style="color:{D['muted']};font-size:11px;">artifact</span>
  </div>
  <div style="padding:16px;overflow-x:auto;white-space:pre;color:#e2e8f0;line-height:1.6;">
{code.replace('<','&lt;').replace('>','&gt;')}
  </div>
</div>
""", unsafe_allow_html=True)

# ── File processing ────────────────────────────────────────────────────────────
def process_uploaded_file(uploaded_file):
    """
    Returns a dict with type and content suitable for Groq messages.
    Supports images (vision) and text-based files.
    """
    fname = uploaded_file.name.lower()
    raw = uploaded_file.read()

    image_exts = (".png", ".jpg", ".jpeg", ".gif", ".webp")
    text_exts   = (".txt", ".md", ".py", ".js", ".ts", ".html", ".css",
                   ".json", ".csv", ".xml", ".yaml", ".yml", ".sh", ".c",
                   ".cpp", ".java", ".rs", ".go", ".rb", ".php", ".swift")

    if any(fname.endswith(e) for e in image_exts):
        b64 = base64.b64encode(raw).decode()
        ext = fname.rsplit(".", 1)[-1]
        mime = f"image/{'jpeg' if ext == 'jpg' else ext}"
        return {"kind": "image", "b64": b64, "mime": mime, "name": uploaded_file.name}

    if any(fname.endswith(e) for e in text_exts):
        try:
            text = raw.decode("utf-8")
        except Exception:
            text = raw.decode("latin-1")
        return {"kind": "text", "content": text, "name": uploaded_file.name}

    # Try decoding any other file as text
    try:
        text = raw.decode("utf-8")
        return {"kind": "text", "content": text, "name": uploaded_file.name}
    except Exception:
        return {"kind": "unsupported", "name": uploaded_file.name}

def build_groq_messages(history, system_prompt, file_info=None, user_text=""):
    """Build Groq-compatible messages list with optional file attachment."""
    msgs = [{"role": "system", "content": system_prompt}]
    msgs.extend(history[-12:])   # context window

    if file_info:
        if file_info["kind"] == "image":
            msgs.append({
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{file_info['mime']};base64,{file_info['b64']}"}},
                    {"type": "text", "text": user_text or f"Describe this image: {file_info['name']}"},
                ]
            })
        elif file_info["kind"] == "text":
            snippet = file_info["content"][:8000]  # stay within token limits
            msgs.append({
                "role": "user",
                "content": f"📎 File attached: `{file_info['name']}`\n\n```\n{snippet}\n```\n\n{user_text}"
            })
        else:
            msgs.append({"role": "user", "content": f"⚠️ Unsupported file: {file_info['name']}\n\n{user_text}"})
    elif user_text:
        msgs.append({"role": "user", "content": user_text})

    return msgs

# ── Session state ──────────────────────────────────────────────────────────────
if "user_id" not in st.session_state:
    # Use a stable cookie-like ID stored in query params if possible, else generate
    st.session_state.user_id = f"helix_{abs(hash(str(time.time())))}"

if "chat_history" not in st.session_state:
    sb = supabase_load(st.session_state.user_id)
    st.session_state.chat_history = sb if sb else load_memory()

if "conversation_title" not in st.session_state:
    st.session_state.conversation_title = "New conversation"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown(f"""
    <div style="padding:20px 16px 12px;border-bottom:1px solid {D['border']};margin-bottom:16px;">
      <div style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;
                  background:linear-gradient(135deg,{D['accent']},{D['accent2']});
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        🧬 HELIX
      </div>
      <div style="font-size:11px;color:{D['muted']};margin-top:2px;letter-spacing:1px;">
        AI ASSISTANT
      </div>
    </div>
    """, unsafe_allow_html=True)

    # New chat
    if st.button("＋  New Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.conversation_title = "New conversation"
        save_memory([])
        supabase_clear(st.session_state.user_id)
        st.rerun()

    st.divider()

    # Theme toggle
    label = "☀️ Light mode" if st.session_state.dark_mode else "🌙 Dark mode"
    if st.button(label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.divider()

    # System stats
    st.markdown(f"<p style='font-size:11px;color:{D['muted']};letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>System</p>", unsafe_allow_html=True)
    st.markdown(f"""
    <div styl
