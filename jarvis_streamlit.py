import os
import json
import requests
import pytz
import time
import math
import re
import operator
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import psutil

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY", "")

if not api_key:
    st.error("❌ OPENROUTER_API_KEY not found!")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

st.set_page_config(
    page_title="HELIX - AI Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if st.session_state.dark_mode:
    bg_color = "#0a0e27"
    text_color = "#ffffff"
    surface_color = "#1a1f3a"
    accent_color = "#4fc3f7"
else:
    bg_color = "#f0f4f8"
    text_color = "#1a1a2e"
    surface_color = "#ffffff"
    accent_color = "#0077b6"

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
            to {{ filter: drop-shadow(0 0 30px {accent_color}); }}
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
        .helix-thinking {{
            display: flex;
            justify-content: left;
            align-items: center;
            padding: 10px;
        }}
        .helix-ring {{
            width: 36px;
            height: 36px;
            border-radius: 50%;
            border: 3px solid transparent;
            border-top: 3px solid #4fc3f7;
            border-right: 3px solid #00e5ff;
            border-bottom: 3px solid #0077ff;
            box-shadow: 0 0 10px #4fc3f7, 0 0 20px #00e5ff, inset 0 0 10px rgba(79, 195, 247, 0.2);
            animation: helixspin 0.8s linear infinite;
        }}
        @keyframes helixspin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
""", unsafe_allow_html=True)

MEMORY_FILE = "jarvis_memory.json"
IST = pytz.timezone('Asia/Kolkata')

# ==========================================
# 🚨 SAFE CALCULATOR - No eval()
# ==========================================
def safe_calculate(expression):
    try:
        expr = expression.lower().strip()
        expr = expr.replace(" ", "")
        expr = expr.replace("^", "**")
        expr = expr.replace("x", "*")

        # Handle sqrt
        sqrt_match = re.search(r'sqrt\(?(\d+\.?\d*)\)?', expr)
        if sqrt_match:
            num = float(sqrt_match.group(1))
            return {"result": round(math.sqrt(num), 6)}

        # Handle basic operations safely
        # Only allow digits and operators
        if re.match(r'^[\d\.\+\-\*\/\(\)\s]+$', expr):
            # Parse manually
            result = safe_eval(expr)
            if result is not None:
                return {"result": round(float(result), 6)}

        # Try trig functions
        sin_match = re.search(r'sin\(?(\d+\.?\d*)\)?', expr)
        cos_match = re.search(r'cos\(?(\d+\.?\d*)\)?', expr)
        tan_match = re.search(r'tan\(?(\d+\.?\d*)\)?', expr)
        log_match = re.search(r'log\(?(\d+\.?\d*)\)?', expr)

        if sin_match:
            return {"result": round(math.sin(math.radians(float(sin_match.group(1)))), 6)}
        if cos_match:
            return {"result": round(math.cos(math.radians(float(cos_match.group(1)))), 6)}
        if tan_match:
            return {"result": round(math.tan(math.radians(float(tan_match.group(1)))), 6)}
        if log_match:
            return {"result": round(math.log10(float(log_match.group(1))), 6)}

        return {"error": "Could not parse expression"}
    except Exception as e:
        return {"error": str(e)}

def safe_eval(expr):
    """Safely evaluate basic math without eval()"""
    try:
        # Remove spaces
        expr = expr.replace(" ", "")

        # Handle parentheses recursively
        while '(' in expr:
            inner = re.search(r'\(([^()]+)\)', expr)
            if inner:
                inner_result = safe_eval(inner.group(1))
                if inner_result is None:
                    return None
                expr = expr[:inner.start()] + str(inner_result) + expr[inner.end():]
            else:
                return None

        # Handle + and - (lowest precedence)
        # Split by + and - but not inside numbers
        tokens = re.split(r'(?<=[0-9])([+\-])(?=[0-9])', expr)
        if len(tokens) > 1:
            result = safe_eval(tokens[0])
            i = 1
            while i < len(tokens):
                op = tokens[i]
                val = safe_eval(tokens[i+1])
                if result is None or val is None:
                    return None
                if op == '+':
                    result += val
                elif op == '-':
                    result -= val
                i += 2
            return result

        # Handle * and /
        tokens = re.split(r'([*\/])', expr)
        if len(tokens) > 1:
            result = float(tokens[0])
            i = 1
            while i < len(tokens):
                op = tokens[i]
                val = float(tokens[i+1])
                if op == '*':
                    result *= val
                elif op == '/':
                    if val == 0:
                        return None
                    result /= val
                i += 2
            return result

        # Handle ** (power)
        if '**' in expr:
            parts = expr.split('**')
            return float(parts[0]) ** float(parts[1])

        return float(expr)
    except:
        return None

# ==========================================
# 🌐 BETTER WEB SEARCH
# ==========================================
def web_search(query):
    try:
        # Try Wikipedia API first for factual queries
        wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        wiki_resp = requests.get(wiki_url, timeout=5)
        if wiki_resp.status_code == 200:
            wiki_data = wiki_resp.json()
            if wiki_data.get('extract'):
                return {"results": [{
                    "title": wiki_data.get('title', query),
                    "snippet": wiki_data.get('extract', '')[:500],
                    "url": wiki_data.get('content_urls', {}).get('desktop', {}).get('page', '')
                }]}

        # Fallback to DuckDuckGo
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "pretty": 1, "no_redirect": 1}
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            results = []
            if data.get('AbstractText'):
                results.append({
                    "title": data.get('Heading', 'Search Result'),
                    "url": data.get('AbstractURL', ''),
                    "snippet": data.get('AbstractText', '')
                })
            if 'RelatedTopics' in data:
                for topic in data['RelatedTopics'][:4]:
                    if 'Text' in topic:
                        results.append({
                            "title": topic.get('Text', '')[:80],
                            "url": topic.get('FirstURL', ''),
                            "snippet": topic.get('Text', '')
                        })
            return {"results": results[:5]} if results else {"error": "No results found"}
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Could not perform web search"}

def get_weather(location="London"):
    try:
        response = requests.get(f"https://wttr.in/{location}?format=j1", timeout=5)
        if response.status_code == 200:
            data = response.json()
            current = data['current_condition'][0]
            return {
                "location": location,
                "temperature": current['temp_C'],
                "description": current['weatherDesc'][0]['value'],
                "humidity": current['humidity'],
                "wind_speed": current['windspeedKmph'],
                "feels_like": current['FeelsLikeC']
            }
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Could not fetch weather data"}

def get_news(query="latest", country="us"):
    try:
        if not news_api_key:
            return {"error": "NEWS_API_KEY not configured"}
        if query == "latest":
            url = f"https://newsapi.org/v2/top-headlines?country={country}&apiKey={news_api_key}"
        else:
            url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={news_api_key}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])[:5]
            news_list = []
            for article in articles:
                news_list.append({
                    "title": article['title'],
                    "source": article['source']['name'],
                    "description": article['description'],
                    "url": article['url'],
                })
            return {"articles": news_list}
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Could not fetch news"}

# ==========================================
# 🤖 AI ROUTING - Smarter detection
# ==========================================
def route_request(user_input):
    user_lower = user_input.lower().strip()

    # Calculator
    math_pattern = re.search(r'[\d]+[\s]*[\+\-\*\/\^][\s]*[\d]+', user_input)
    sqrt_pattern = re.search(r'(square root of|sqrt)\s*[\d]+', user_lower)
    trig_pattern = re.search(r'(sin|cos|tan|log)\s*\(?[\d]+', user_lower)
    if math_pattern or sqrt_pattern or trig_pattern or re.search(r'^(calculate|compute|what is)\s+[\d]', user_lower):
        if sqrt_pattern:
            num = re.search(r'[\d]+', sqrt_pattern.group())
            if num:
                return {"type": "calculator", "expression": f"sqrt({num.group()})"}
        if trig_pattern:
            return {"type": "calculator", "expression": trig_pattern.group()}
        if math_pattern:
            return {"type": "calculator", "expression": math_pattern.group().strip()}

    # Weather
    if any(word in user_lower for word in ["weather", "temperature", "forecast", "climate", "raining", "snowing", "humid"]):
        location = "London"
        loc_match = re.search(r'\bin\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)', user_lower)
        if loc_match:
            location = loc_match.group(1).title()
        return {"type": "weather", "location": location}

    # News
    if any(word in user_lower for word in ["news", "headlines", "latest news", "breaking news"]):
        query = "latest"
        for word in ["about", "regarding", "on"]:
            if word in user_lower:
                parts = user_lower.split(word, 1)
                if len(parts) > 1:
                    query = parts[1].strip()
                    break
        return {"type": "news", "query": query}

    # Web search
    if any(phrase in user_lower for phrase in ["search for", "look up", "google", "search the web"]):
        search_query = user_input
        for phrase in ["search for", "search", "look up", "google"]:
            if phrase in user_lower:
                search_query = user_input.split(phrase, 1)[1].strip()
                break
        return {"type": "search", "query": search_query}

    return None

def auto_web_search_needed(response_text):
    uncertainty_phrases = [
        "i don't know", "i'm not sure", "i cannot find",
        "i don't have information", "beyond my knowledge",
        "i'm unable to", "not in my knowledge", "i lack information",
        "i do not have", "cannot recall", "not aware of",
        "i have no information"
    ]
    return any(phrase in response_text.lower() for phrase in uncertainty_phrases)

# ==========================================
# 🧠 BETTER MEMORY
# ==========================================
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(chat_history):
    with open(MEMORY_FILE, "w") as f:
        json.dump(chat_history, f, indent=2)

def get_smart_context(chat_history, max_messages=20):
    """Get smart context — keep recent + important messages"""
    if len(chat_history) <= max_messages:
        return chat_history
    # Always keep last 10 messages + first 5 for context
    return chat_history[:5] + chat_history[-15:]

def show_thinking(placeholder):
    placeholder.markdown("""
    <div class='helix-thinking'>
        <div class='helix-ring'></div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# ⚡ STREAMING - Word by word response
# ==========================================
def stream_response(messages, placeholder):
    """Stream response word by word"""
    try:
        stream = client.chat.completions.create(
            extra_headers={"HTTP-Referer": "http://localhost", "X-Title": "Helix"},
            model="openrouter/auto",
            messages=messages,
            stream=True
        )
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                placeholder.markdown(f"**🧬 HELIX:** {full_response}▌")
        placeholder.markdown(f"**🧬 HELIX:** {full_response}")
        return full_response
    except Exception:
        # Fallback to non-streaming
        completion = client.chat.completions.create(
            extra_headers={"HTTP-Referer": "http://localhost", "X-Title": "Helix"},
            model="openrouter/auto",
            messages=messages
        )
        response = completion.choices[0].message.content
        placeholder.markdown(f"**🧬 HELIX:** {response}")
        return response

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_memory()

st.markdown(f"""
<div class='helix-avatar'>
    <div class='helix-logo'>🧬</div>
    <h1 style='color:{accent_color}; margin:0; font-size:36px; letter-spacing:4px;'>HELIX</h1>
    <p style='color:{accent_color}; font-family: monospace; margin:5px 0;'>▓▓▓ MEMORY ONLINE ▓▓▓</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    mode_label = "☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode"
    if st.button(mode_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    st.divider()
    st.markdown("### SYSTEM STATUS")
    st.write(f"🕐 {datetime.now(IST).strftime('%H:%M:%S IST')}")
    st.write(f"💾 RAM: {psutil.virtual_memory().percent}%")
    st.write(f"⚙️ CPU: {psutil.cpu_percent()}%")
    st.divider()
    total = len(st.session_state.chat_history)
    st.write(f"📊 Total Messages: {total}")
    if st.button("🗑️ Clear Memory", use_container_width=True):
        st.session_state.chat_history = []
        save_memory([])
        st.rerun()
    st.divider()
    st.markdown("### FEATURES")
    st.markdown("🌤️ **Weather** - Ask about weather\n\n🗞️ **News** - Get latest headlines\n\n🔍 **Web Search** - Search the web\n\n🧮 **Calculator** - Solve math\n\n🔎 **Auto Search** - Searches when unsure\n\n⚡ **Streaming** - Real time responses")

for msg in st.session_state.chat_history[-20:]:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        role = "👤 SIR" if msg["role"] == "user" else "🧬 HELIX"
        st.markdown(f"**{role}:** {msg['content']}")

user_input = st.chat_input("Speak or type, Sir...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    save_memory(st.session_state.chat_history)
    with st.chat_message("user"):
        st.markdown(f"**👤 SIR:** {user_input}")
    with st.chat_message("assistant"):
        try:
            routed = route_request(user_input)
            response = None
            thinking_placeholder = st.empty()

            if routed:
                show_thinking(thinking_placeholder)

                if routed["type"] == "calculator":
                    calc_result = safe_calculate(routed["expression"])
                    thinking_placeholder.empty()
                    if "result" in calc_result:
                        response = f"🧮 **Result:** `{routed['expression']}` = **{calc_result['result']}**"
                    else:
                        response = f"I couldn't calculate that, Sir: {calc_result.get('error', 'Unknown error')}"

                elif routed["type"] == "weather":
                    weather_data = get_weather(routed["location"])
                    thinking_placeholder.empty()
                    if "error" not in weather_data:
                        response = f"""🌤️ **Weather in {weather_data['location']}:**
- 🌡️ Temperature: {weather_data['temperature']}°C (Feels like {weather_data['feels_like']}°C)
- 📝 Condition: {weather_data['description']}
- 💧 Humidity: {weather_data['humidity']}%
- 💨 Wind Speed: {weather_data['wind_speed']} km/h"""
                    else:
                        response = f"Couldn't fetch weather, Sir: {weather_data['error']}"

                elif routed["type"] == "news":
                    news_data = get_news(routed["query"])
                    thinking_placeholder.empty()
                    if "articles" in news_data:
                        response = "🗞️ **Latest News Headlines:**\n\n"
                        for i, article in enumerate(news_data["articles"], 1):
                            response += f"{i}. **{article['title']}**\n   Source: {article['source']}\n   {article['description']}\n   [Read more]({article['url']})\n\n"
                    else:
                        response = f"Couldn't fetch news, Sir: {news_data.get('error', 'Unknown error')}"

                elif routed["type"] == "search":
                    search_data = web_search(routed["query"])
                    thinking_placeholder.empty()
                    if "results" in search_data:
                        response = f"🔍 **Search Results for '{routed['query']}':**\n\n"
                        for i, result in enumerate(search_data["results"], 1):
                            if result.get('url'):
                                response += f"{i}. **{result['title']}**\n   {result['snippet'][:200]}\n   [Read more]({result['url']})\n\n"
                            else:
                                response += f"{i}. {result['snippet'][:200]}\n\n"
                    else:
                        response = f"Couldn't find results, Sir: {search_data.get('error', 'Unknown error')}"

            if response is None:
                show_thinking(thinking_placeholder)
                current_time = datetime.now(IST)
                system_prompt = f"""You are HELIX, an advanced AI assistant. Follow these rules strictly:
1. Be witty and British in tone
2. Call the user Sir
3. NEVER mention the current time or date unless the user specifically asks
4. NEVER mention your creator's name unless the user specifically asks "who created you" or "who made you"
5. Keep responses clean, concise and helpful
6. Do not end responses with unnecessary remarks about system updates or your own status
Today is {current_time.strftime('%A, %d %B %Y')} and time is {current_time.strftime('%I:%M %p')} IST — only use this if asked."""

                context = get_smart_context(st.session_state.chat_history)
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(context[:-1])  # exclude last user message as it's already in context

                thinking_placeholder.empty()
                response_placeholder = st.empty()
                response = stream_response(messages + [{"role": "user", "content": user_input}], response_placeholder)

                if auto_web_search_needed(response):
                    show_thinking(thinking_placeholder)
                    search_data = web_search(user_input)
                    if "results" in search_data and search_data["results"]:
                        search_context = "\n".join([r['snippet'] for r in search_data["results"][:3]])
                        followup = [{"role": "system", "content": system_prompt}]
                        followup.append({"role": "user", "content": user_input})
                        followup.append({"role": "assistant", "content": response})
                        followup.append({"role": "user", "content": f"Web search found: {search_context}\n\nGive a better answer using this."})
                        thinking_placeholder.empty()
                        response = stream_response(followup, response_placeholder)
                        response = "🔎 *(Web searched)*\n\n" + response
                        response_placeholder.markdown(f"**🧬 HELIX:** {response}")
                    thinking_placeholder.empty()

            if not (response is None):
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                save_memory(st.session_state.chat_history)

                if routed:
                    result_placeholder = st.empty()
                    result_placeholder.markdown(f"**🧬 HELIX:** {response}")

        except Exception as e:
            st.error(f"SYSTEM ERROR: {str(e)}")
    st.rerun()
