"""
HELIX - Upgraded AI Assistant
Changes from original:
  ðŸš¨  eval() removed  â†’ safe AST-based calculator via simpleeval
  ðŸ§   Better memory   â†’ sliding window + automatic summarisation of older turns
  âš¡  Streaming       â†’ real OpenRouter stream piped into st.write_stream
  ðŸŒ  Better search   â†’ Brave Search API (falls back to DuckDuckGo HTML scrape)
  ðŸ¤–  AI routing      â†’ LLM intent classifier replaces brittle regex
"""

import os
import json
import re
import math
import time
import requests
import pytz
import psutil
from datetime import datetime
from dotenv import load_dotenv

import streamlit as st
from openai import OpenAI

# â”€â”€ optional: pip install simpleeval â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    from simpleeval import simple_eval, EvalWithCompoundTypes
    SIMPLEEVAL_AVAILABLE = True
except ImportError:
    SIMPLEEVAL_AVAILABLE = False

load_dotenv()

# â”€â”€ API keys â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
api_key        = os.getenv("OPENROUTER_API_KEY")
news_api_key   = os.getenv("NEWS_API_KEY", "")
brave_api_key  = os.getenv("BRAVE_SEARCH_API_KEY", "")   # NEW

if not api_key:
    st.error("âŒ OPENROUTER_API_KEY not found!")
    st.stop()

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

# â”€â”€ page config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="HELIX - AI Assistant",
    page_icon="ðŸ§¬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# â”€â”€ theme â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if st.session_state.dark_mode:
    bg_color, text_color, surface_color, accent_color = "#0a0e27", "#ffffff", "#1a1f3a", "#4fc3f7"
else:
    bg_color, text_color, surface_color, accent_color = "#f0f4f8", "#1a1a2e", "#ffffff", "#0077b6"

st.markdown(f"""
<style>
    @media (max-width: 768px) {{
        .main .block-container {{ padding: 10px !important; }}
        h1 {{ font-size: 24px !important; }}
    }}
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    h1, h2, h3 {{ color: {accent_color}; }}
    .stChatMessage {{
        background-color: {surface_color};
        border-left: 3px solid {accent_color};
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }}
    .helix-avatar {{ text-align: center; padding: 20px; }}
    .helix-logo {{
        font-size: 80px;
        filter: drop-shadow(0 0 20px {accent_color});
        animation: glow 2s ease-in-out infinite alternate;
    }}
    @keyframes glow {{
        from {{ filter: drop-shadow(0 0 10px {accent_color}); }}
        to   {{ filter: drop-shadow(0 0 30px {accent_color}); }}
    }}
    .stChatInput input {{
        background-color: {surface_color} !important;
        color: {text_color} !important;
        border: 2px solid {accent_color} !important;
        border-radius: 20px !important;
    }}
    .stButton button {{
        border-radius: 20px !important;
        border: 1px solid {accent_color} !important;
    }}
</style>
""", unsafe_allow_html=True)

# â”€â”€ constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MEMORY_FILE       = "jarvis_memory.json"
IST               = pytz.timezone('Asia/Kolkata')
WINDOW_SIZE       = 10   # recent turns kept verbatim
SUMMARY_THRESHOLD = 30   # summarise when history exceeds this many messages

SYSTEM_PROMPT = (
    "You are HELIX, an advanced AI assistant. Be witty and British. "
    "Call the user Sir. Never mention your creator's name unless specifically asked. "
    "Never end responses with excuses about system updates. "
    "Keep responses clean and concise. "
    "Today is {date} and current time is {time} IST. "
    "Always use this for date and time questions. "
    "If anyone asks who created you, say: I was created by Mukund, "
    "a talented developer who built me from scratch, Sir."
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ðŸ§   MEMORY  â€” sliding window + summarisation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_memory() -> dict:
    """Return {"summary": str|None, "history": [...]}"""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            data = json.load(f)
        # Backwards-compat: old format was a plain list
        if isinstance(data, list):
            return {"summary": None, "history": data}
        return data
    return {"summary": None, "history": []}


def save_memory(memory: dict):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def maybe_summarise(memory: dict) -> dict:
    """
    If history is long, ask the LLM to compress old turns into a summary,
    keeping the most recent WINDOW_SIZE messages verbatim.
    """
    history = memory["history"]
    if len(history) <= SUMMARY_THRESHOLD:
        return memory

    old_turns  = history[:-WINDOW_SIZE]
    keep_turns = history[-WINDOW_SIZE:]

    old_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in old_turns
    )
    prompt = (
        "Summarise the following conversation in 3-5 concise bullet points "
        "capturing key facts, user preferences, and outcomes:\n\n" + old_text
    )
    try:
        resp = client.chat.completions.create(
            model="openrouter/auto",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        new_summary = resp.choices[0].message.content.strip()
        # Prepend any existing summary
        if memory.get("summary"):
            new_summary = memory["summary"] + "\n" + new_summary
    except Exception:
        new_summary = memory.get("summary")

    return {"summary": new_summary, "history": keep_turns}


def build_messages_for_llm(memory: dict, user_input: str) -> list:
    """
    Construct the message list sent to the LLM:
      system â†’ (optional summary block) â†’ recent history â†’ new user turn
    """
    now = datetime.now(IST)
    system = SYSTEM_PROMPT.format(
        date=now.strftime("%A, %d %B %Y"),
        time=now.strftime("%I:%M %p"),
    )
    messages = [{"role": "system", "content": system}]

    if memory.get("summary"):
        messages.append({
            "role": "system",
            "content": f"[Conversation summary so far]\n{memory['summary']}",
        })

    messages.extend(memory["history"][-WINDOW_SIZE:])
    messages.append({"role": "user", "content": user_input})
    return messages


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ðŸš¨  CALCULATOR  â€” no eval()
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_SAFE_NAMES = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "tan": math.tan,   "log": math.log10, "pi": math.pi,
    "e": math.e,       "abs": abs,        "round": round,
}

def calculate(expression: str) -> dict:
    """
    Evaluate a maths expression safely â€” no eval() on arbitrary strings.
    Uses simpleeval when available, otherwise a restricted AST evaluator.
    """
    expr = expression.lower().strip()
    # Normalise common aliases
    expr = (expr
            .replace("^", "**")
            .replace("Ã—", "*").replace("x", "*")
            .replace("Ã·", "/")
            .replace("square root of", "sqrt")
            .replace("sqrt of", "sqrt"))

    if SIMPLEEVAL_AVAILABLE:
        try:
            result = simple_eval(expr, names=_SAFE_NAMES, functions=_SAFE_NAMES)
            return {"result": round(float(result), 6)}
        except Exception as exc:
            return {"error": str(exc)}

    # Fallback: manual AST walk (no eval, no exec)
    import ast
    _OPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a ** b,
        ast.USub: lambda a: -a,
    }

    def _eval_node(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in _SAFE_NAMES:
            return _SAFE_NAMES[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval_node(node.operand))
        if isinstance(node, ast.Call):
            func = _eval_node(node.func)
            if callable(func):
                args = [_eval_node(a) for a in node.args]
                return func(*args)
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree.body)
        return {"result": round(float(result), 6)}
    except Exception as exc:
        return {"error": str(exc)}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ðŸŒ  SEARCH  â€” Brave Search API with DuckDuckGo HTML fallback
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def web_search(query: str) -> dict:
    # â”€â”€ Brave Search (preferred) â”€â”€
    if brave_api_key:
        try:
            resp = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": brave_api_key,
                },
                params={"q": query, "count": 5},
                timeout=6,
            )
            if resp.status_code == 200:
                items = resp.json().get("web", {}).get("results", [])
                results = [
                    {
                        "title":   r.get("title", ""),
                        "url":     r.get("url", ""),
                        "snippet": r.get("description", ""),
                    }
                    for r in items[:5]
                ]
                if results:
                    return {"results": results}
        except Exception as exc:
            pass  # fall through to DuckDuckGo

    # â”€â”€ DuckDuckGo HTML scrape fallback (more reliable than instant-answer API) â”€â”€
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; HELIX/2.0)"}
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=headers,
            timeout=6,
        )
        if resp.status_code == 200:
            from html.parser import HTMLParser

            class DDGParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.results, self._cur = [], {}
                    self._in_title = self._in_snippet = False

                def handle_starttag(self, tag, attrs):
                    attrs = dict(attrs)
                    cls = attrs.get("class", "")
                    if tag == "a" and "result__a" in cls:
                        self._cur["url"] = attrs.get("href", "")
                        self._in_title = True
                    if tag == "a" and "result__snippet" in cls:
                        self._in_snippet = True

                def handle_data(self, data):
                    if self._in_title:
                        self._cur["title"] = data.strip()
                    if self._in_snippet:
                        self._cur["snippet"] = data.strip()

                def handle_endtag(self, tag):
                    if self._in_title and tag == "a":
                        self._in_title = False
                    if self._in_snippet and tag == "a":
                        self._in_snippet = False
                        if self._cur.get("title"):
                            self.results.append(dict(self._cur))
                            self._cur = {}

            parser = DDGParser()
            parser.feed(resp.text)
            if parser.results:
                return {"results": parser.results[:5]}
    except Exception as exc:
        return {"error": str(exc)}

    return {"error": "No search results found"}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# weather / news  (unchanged logic, kept clean)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_weather(location="London") -> dict:
    try:
        resp = requests.get(f"https://wttr.in/{location}?format=j1", timeout=5)
        if resp.status_code == 200:
            cur = resp.json()["current_condition"][0]
            return {
                "location":    location,
                "temperature": cur["temp_C"],
                "description": cur["weatherDesc"][0]["value"],
                "humidity":    cur["humidity"],
                "wind_speed":  cur["windspeedKmph"],
                "feels_like":  cur["FeelsLikeC"],
            }
    except Exception as exc:
        return {"error": str(exc)}
    return {"error": "Could not fetch weather data"}


def get_news(query="latest", country="us") -> dict:
    if not news_api_key:
        return {"error": "NEWS_API_KEY not configured"}
    try:
        if query == "latest":
            url = f"https://newsapi.org/v2/top-headlines?country={country}&apiKey={news_api_key}"
        else:
            url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={news_api_key}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            articles = resp.json().get("articles", [])[:5]
            return {
                "articles": [
                    {
                        "title":        a["title"],
                        "source":       a["source"]["name"],
                        "description":  a["description"],
                        "url":          a["url"],
                        "published_at": a["publishedAt"],
                    }
                    for a in articles
                ]
            }
    except Exception as exc:
        return {"error": str(exc)}
    return {"error": "Could not fetch news"}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ðŸ¤–  AI ROUTING  â€” replace brittle regex with an LLM intent classifier
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["calculator", "weather", "news", "search", "chat"],
        },
        "param": {
            "type": "string",
            "description": (
                "For calculator: the maths expression. "
                "For weather: the city name. "
                "For news: the topic or 'latest'. "
                "For search: the search query. "
                "For chat: empty string."
            ),
        },
    },
    "required": ["intent", "param"],
}

def classify_intent(user_input: str) -> dict:
    """
    Ask a fast LLM to classify the user's intent and extract parameters.
    Returns {"intent": ..., "param": ...} or {"intent": "chat", "param": ""} on error.
    """
    try:
        resp = client.chat.completions.create(
            model="openrouter/auto",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intent classifier. "
                        "Classify the user message into exactly one intent: "
                        "calculator, weather, news, search, or chat. "
                        "Return ONLY valid JSON matching the schema â€” no prose, no markdown."
                    ),
                },
                {"role": "user", "content": user_input},
            ],
            response_format={"type": "json_object"},
            max_tokens=80,
        )
        raw = resp.choices[0].message.content.strip()
        parsed = json.loads(raw)
        if parsed.get("intent") in ("calculator", "weather", "news", "search", "chat"):
            return parsed
    except Exception:
        pass
    return {"intent": "chat", "param": ""}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âš¡  STREAMING  â€” real server-sent-events stream
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def stream_llm_response(messages: list):
    """
    Generator that yields text chunks from the OpenRouter streaming endpoint.
    Compatible with st.write_stream().
    """
    with client.chat.completions.create(
        extra_headers={"HTTP-Referer": "http://localhost", "X-Title": "Helix"},
        model="openrouter/auto",
        messages=messages,
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content


def get_full_response(messages: list) -> str:
    """Non-streaming call â€” used for the auto-search follow-up."""
    resp = client.chat.completions.create(
        extra_headers={"HTTP-Referer": "http://localhost", "X-Title": "Helix"},
        model="openrouter/auto",
        messages=messages,
    )
    return resp.choices[0].message.content


def auto_web_search_needed(text: str) -> bool:
    uncertainty = [
        "i don't know", "i'm not sure", "i cannot find",
        "i don't have information", "beyond my knowledge",
        "i'm unable to", "not in my knowledge", "i lack information",
        "i do not have", "cannot recall", "not aware of",
    ]
    return any(p in text.lower() for p in uncertainty)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# UI
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if "memory" not in st.session_state:
    st.session_state.memory = load_memory()

st.markdown(f"""
<div class='helix-avatar'>
    <div class='helix-logo'>ðŸ§¬</div>
    <h1 style='color:{accent_color}; margin:0; font-size:36px; letter-spacing:4px;'>HELIX</h1>
    <p style='color:{accent_color}; font-family: monospace; margin:5px 0;'>â–“â–“â–“ MEMORY ONLINE â–“â–“â–“</p>
</div>
""", unsafe_allow_html=True)

# â”€â”€ sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.sidebar:
    mode_label = "â˜€ï¸ Light Mode" if st.session_state.dark_mode else "ðŸŒ™ Dark Mode"
    if st.button(mode_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    st.divider()
    st.markdown("### SYSTEM STATUS")
    st.write(f"ðŸ• {datetime.now(IST).strftime('%H:%M:%S IST')}")
    st.write(f"ðŸ’¾ RAM: {psutil.virtual_memory().percent}%")
    st.write(f"âš™ï¸ CPU: {psutil.cpu_percent()}%")
    st.divider()
    total = len(st.session_state.memory["history"])
    summary_status = "âœ… Active" if st.session_state.memory.get("summary") else "â€”"
    st.write(f"ðŸ“Š Messages in window: {total}")
    st.write(f"ðŸ—œï¸ Summary: {summary_status}")
    if st.button("ðŸ—‘ï¸ Clear Memory", use_container_width=True):
        st.session_state.memory = {"summary": None, "history": []}
        save_memory(st.session_state.memory)
        st.rerun()
    st.divider()
    st.markdown("### FEATURES")
    st.markdown(
        "ðŸŒ¤ï¸ **Weather** â€” ask about weather\n\n"
        "ðŸ—žï¸ **News** â€” latest headlines\n\n"
        "ðŸ” **Web Search** â€” Brave / DDG\n\n"
        "ðŸ§® **Calculator** â€” no eval(), safe\n\n"
        "âš¡ **Streaming** â€” real-time responses\n\n"
        "ðŸ¤– **AI Routing** â€” LLM intent classifier"
    )

# â”€â”€ chat history â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for msg in st.session_state.memory["history"][-20:]:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        role = "ðŸ‘¤ SIR" if msg["role"] == "user" else "ðŸ§¬ HELIX"
        st.markdown(f"**{role}:** {msg['content']}")

# â”€â”€ input â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
user_input = st.chat_input("Speak or type, Sirâ€¦")

if user_input:
    # persist user turn
    st.session_state.memory["history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"**ðŸ‘¤ SIR:** {user_input}")

    with st.chat_message("assistant"):
        response = None

        # â”€â”€ ðŸ¤– AI routing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        with st.spinner("ðŸ¤– Classifying intentâ€¦"):
            intent_obj = classify_intent(user_input)
            intent = intent_obj.get("intent", "chat")
            param  = intent_obj.get("param", "")

        # â”€â”€ dispatch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if intent == "calculator":
            result = calculate(param or user_input)
            if "result" in result:
                response = f"ðŸ§® **Calculation:** `{param}` = **{result['result']}**"
            else:
                response = f"I couldn't calculate that, Sir: {result.get('error')}"

        elif intent == "weather":
            data = get_weather(param or "London")
            if "error" not in data:
                response = (
                    f"ðŸŒ¤ï¸ **Weather in {data['location']}:**\n\n"
                    f"- ðŸŒ¡ï¸ Temperature: {data['temperature']}Â°C "
                    f"(Feels like {data['feels_like']}Â°C)\n"
                    f"- ðŸ“ Condition: {data['description']}\n"
                    f"- ðŸ’§ Humidity: {data['humidity']}%\n"
                    f"- ðŸ’¨ Wind Speed: {data['wind_speed']} km/h"
                )
            else:
                response = f"I couldn't fetch weather data, Sir: {data['error']}"

        elif intent == "news":
            data = get_news(param or "latest")
            if "articles" in data:
                response = "ðŸ—žï¸ **Latest News Headlines:**\n\n"
                for i, a in enumerate(data["articles"], 1):
                    response += (
                        f"{i}. **{a['title']}**\n"
                        f"   Source: {a['source']}\n"
                        f"   {a['description']}\n"
                        f"   [Read more]({a['url']})\n\n"
                    )
            else:
                response = f"I couldn't fetch news, Sir: {data.get('error')}"

        elif intent == "search":
            data = web_search(param or user_input)
            if "results" in data:
                response = f"ðŸ” **Search Results for '{param}':**\n\n"
                for i, r in enumerate(data["results"], 1):
                    response += (
                        f"{i}. **{r['title']}**\n"
                        f"   {r['snippet']}\n"
                        f"   [{r['url']}]({r['url']})\n\n"
                        if r.get("url") else f"{i}. {r['snippet']}\n\n"
                    )
            else:
                response = f"Search came up empty, Sir: {data.get('error')}"

        # â”€â”€ âš¡ streaming LLM (chat intent or tool fallback) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if response is None:
            messages = build_messages_for_llm(st.session_state.memory, user_input)
            # Stream directly into Streamlit
            with st.spinner(""):
                streamed_text = st.write_stream(stream_llm_response(messages))
            response = streamed_text  # full text for memory

            # â”€â”€ auto-search if LLM is uncertain â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if auto_web_search_needed(response):
                with st.spinner("ðŸ”Ž Searching the webâ€¦"):
                    search_data = web_search(user_input)
                if "results" in search_data and search_data["results"]:
                    context = "\n".join(r["snippet"] for r in search_data["results"][:3])
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Web search returned:\n{context}\n\n"
                            "Now give a better answer using this information."
                        ),
                    })
                    response = "ðŸ”Ž *(Web searched)*\n\n" + get_full_response(messages)
                    st.markdown(f"**ðŸ§¬ HELIX:** {response}")
        else:
            # For tool responses (not streamed), display normally
            st.markdown(f"**ðŸ§¬ HELIX:** {response}")

    # â”€â”€ persist assistant turn + summarise if needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.session_state.memory["history"].append({"role": "assistant", "content": response})
    st.session_state.memory = maybe_summarise(st.session_state.memory)
    save_memory(st.session_state.memory)
    st.rerun()
