import os
import json
import requests
import pytz
import time
import math
import re
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
        .thinking-container {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px;
        }}
        .thinking-star {{
            width: 28px;
            height: 28px;
            color: {accent_color};
            animation: spin 1.2s linear infinite;
            font-size: 24px;
            display: inline-block;
            filter: drop-shadow(0 0 8px {accent_color});
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        .thinking-text {{
            color: {accent_color};
            font-family: monospace;
            font-size: 14px;
            animation: pulse 1.5s ease-in-out infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.4; }}
        }}
        .thinking-dots {{
            display: inline-block;
            animation: dots 1.5s steps(4, end) infinite;
        }}
        @keyframes dots {{
            0%, 20% {{ content: '.'; }}
            40% {{ content: '..'; }}
            60% {{ content: '...'; }}
            80%, 100% {{ content: ''; }}
        }}
    </style>
""", unsafe_allow_html=True)

MEMORY_FILE = "jarvis_memory.json"
IST = pytz.timezone('Asia/Kolkata')

def show_thinking(placeholder, text="HELIX is thinking"):
    placeholder.markdown(f"""
    <div class='thinking-container'>
        <span class='thinking-star'>✳️</span>
        <span class='thinking-text'>{text}<span class='thinking-dots'>...</span></span>
    </div>
    """, unsafe_allow_html=True)

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
                    "published_at": article['publishedAt']
                })
            return {"articles": news_list}
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Could not fetch news"}

def web_search(query):
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "pretty": 1, "no_redirect": 1}
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            results = []
            if 'RelatedTopics' in data:
                for topic in data['RelatedTopics'][:5]:
                    if 'Text' in topic:
                        results.append({
                            "title": topic.get('Text', ''),
                            "url": topic.get('FirstURL', ''),
                            "snippet": topic.get('Text', '')
                        })
            if data.get('AbstractText'):
                results.insert(0, {
                    "title": data.get('Heading', 'Search Result'),
                    "url": data.get('AbstractURL', ''),
                    "snippet": data.get('AbstractText', '')
                })
            return {"results": results[:5]} if results else {"error": "No results found"}
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Could not perform web search"}

def calculate(expression):
    try:
        expr = expression.lower().strip()
        expr = expr.replace("x", "*").replace("^", "**")
        expr = expr.replace("pi", str(math.pi))
        expr = expr.replace("square root of", "math.sqrt")
        expr = expr.replace("sqrt of", "math.sqrt")
        expr = expr.replace("sqrt", "math.sqrt")
        expr = expr.replace("sin", "math.sin")
        expr = expr.replace("cos", "math.cos")
        expr = expr.replace("tan", "math.tan")
        expr = expr.replace("log", "math.log10")
        expr = re.sub(r'math\.sqrt\s+(\d+)', r'math.sqrt(\1)', expr)
        expr = re.sub(r'math\.sqrt\s*\((\d+)\)', r'math.sqrt(\1)', expr)
        allowed = re.sub(r'[0-9\s\+\-\*\/\.\(\)e]', '',
                  expr.replace("math.sqrt", "")
                      .replace("math.sin", "")
                      .replace("math.cos", "")
                      .replace("math.tan", "")
                      .replace("math.log10", "")
                      .replace("math.pi", ""))
        if allowed == "":
            result = eval(expr, {"__builtins__": {}}, {"math": math})
            return {"result": round(float(result), 6)}
        return {"error": "Invalid expression"}
    except Exception as e:
        return {"error": str(e)}

def check_for_special_requests(user_input):
    user_lower = user_input.lower()
    math_pattern = re.search(r'[\d]+[\s]*[\+\-\*\/\^][\s]*[\d]+', user_input)
    sqrt_pattern = re.search(r'(square root of|sqrt\s+of|sqrt)\s*[\d]+', user_lower)
    if math_pattern or sqrt_pattern or any(word in user_lower for word in ["calculate", "compute"]):
        if sqrt_pattern:
            num = re.search(r'[\d]+', sqrt_pattern.group())
            if num:
                return {"type": "calculator", "expression": f"math.sqrt({num.group()})"}
        if math_pattern:
            return {"type": "calculator", "expression": math_pattern.group().strip()}
        expr = re.search(r'[\d\s\+\-\*\/\.\(\)\^]+', user_input)
        if expr:
            return {"type": "calculator", "expression": expr.group().strip()}

    if any(word in user_lower for word in ["weather", "temperature", "forecast", "climate", "rain", "snow"]):
        location = "London"
        if "in " in user_lower:
            parts = user_lower.split("in ")
            if len(parts) > 1:
                location = parts[1].split()[0].capitalize()
        return {"type": "weather", "location": location}

    if any(word in user_lower for word in ["news", "headlines", "latest news", "breaking"]):
        query = "latest"
        for word in ["about", "regarding", "on", "for"]:
            if word in user_lower:
                parts = user_lower.split(word)
                if len(parts) > 1:
                    query = parts[1].strip()
                    break
        return {"type": "news", "query": query}

    if any(word in user_lower for word in ["search", "find", "look up", "google", "web", "lookup"]):
        search_query = user_input
        for phrase in ["search for", "search", "find", "look up", "lookup"]:
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
        "i do not have", "cannot recall", "not aware of"
    ]
    return any(phrase in response_text.lower() for phrase in uncertainty_phrases)

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(chat_history):
    with open(MEMORY_FILE, "w") as f:
        json.dump(chat_history, f, indent=2)

def type_text(text, placeholder):
    typed = ""
    for char in text:
        typed += char
        placeholder.markdown(f"**🧬 HELIX:** {typed}▌")
        time.sleep(0.01)
    placeholder.markdown(f"**🧬 HELIX:** {typed}")

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
    st.markdown("🌤️ **Weather** - Ask about weather\n\n🗞️ **News** - Get latest headlines\n\n🔍 **Web Search** - Search the web\n\n🧮 **Calculator** - Solve math\n\n🔎 **Auto Search** - Searches when unsure")

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
            special_request = check_for_special_requests(user_input)
            response = None
            thinking_placeholder = st.empty()

            if special_request:
                show_thinking(thinking_placeholder, "HELIX is processing")
                if special_request["type"] == "calculator":
                    calc_result = calculate(special_request["expression"])
                    thinking_placeholder.empty()
                    if "result" in calc_result:
                        response = f"🧮 **Calculation Result:**\n\n`{special_request['expression']}` = **{calc_result['result']}**"
                    else:
                        response = f"I couldn't calculate that: {calc_result.get('error', 'Unknown error')}"
                elif special_request["type"] == "weather":
                    show_thinking(thinking_placeholder, "Fetching weather data")
                    weather_data = get_weather(special_request["location"])
                    thinking_placeholder.empty()
                    if "error" not in weather_data:
                        response = f"""🌤️ **Weather in {weather_data['location']}:**
- 🌡️ Temperature: {weather_data['temperature']}°C (Feels like {weather_data['feels_like']}°C)
- 📝 Condition: {weather_data['description']}
- 💧 Humidity: {weather_data['humidity']}%
- 💨 Wind Speed: {weather_data['wind_speed']} km/h"""
                    else:
                        response = f"I couldn't fetch weather data: {weather_data['error']}"
                elif special_request["type"] == "news":
                    show_thinking(thinking_placeholder, "Fetching latest news")
                    news_data = get_news(special_request["query"])
                    thinking_placeholder.empty()
                    if "articles" in news_data:
                        response = "🗞️ **Latest News Headlines:**\n\n"
                        for i, article in enumerate(news_data["articles"], 1):
                            response += f"{i}. **{article['title']}**\n   Source: {article['source']}\n   {article['description']}\n   [Read more]({article['url']})\n\n"
                    else:
                        response = f"I couldn't fetch news: {news_data.get('error', 'Unknown error')}"
                elif special_request["type"] == "search":
                    show_thinking(thinking_placeholder, "Searching the web")
                    search_data = web_search(special_request["query"])
                    thinking_placeholder.empty()
                    if "results" in search_data:
                        response = f"🔍 **Search Results for '{special_request['query']}':**\n\n"
                        for i, result in enumerate(search_data["results"], 1):
                            if result['url']:
                                response += f"{i}. **{result['title']}**\n   [{result['url']}]({result['url']})\n\n"
                            else:
                                response += f"{i}. {result['snippet']}\n\n"
                    else:
                        response = f"I couldn't find search results: {search_data.get('error', 'Unknown error')}"

            if response is None:
                show_thinking(thinking_placeholder, "HELIX is thinking")
                current_time = datetime.now(IST)
                messages = [{"role": "system", "content": f"You are HELIX, an advanced AI assistant. Be witty and British. Call the user Sir. Never mention your creator's name unless specifically asked. Never end responses with excuses about system updates. Keep responses clean and concise. Today is {current_time.strftime('%A, %d %B %Y')} and current time is {current_time.strftime('%I:%M %p')} IST. Always use this for date and time questions. If anyone asks who created you, say: I was created by Mukund, a talented developer who built me from scratch, Sir."}]
                messages.extend(st.session_state.chat_history[-10:])
                completion = client.chat.completions.create(
                    extra_headers={"HTTP-Referer": "http://localhost", "X-Title": "Helix"},
                    model="openrouter/auto",
                    messages=messages
                )
                response = completion.choices[0].message.content

                if auto_web_search_needed(response):
                    show_thinking(thinking_placeholder, "Searching the web for better answer")
                    search_data = web_search(user_input)
                    if "results" in search_data and search_data["results"]:
                        search_context = "\n".join([r['snippet'] for r in search_data["results"][:3]])
                        messages.append({"role": "assistant", "content": response})
                        messages.append({"role": "user", "content": f"I found this from web search: {search_context}\n\nNow give a better answer based on this."})
                        completion2 = client.chat.completions.create(
                            extra_headers={"HTTP-Referer": "http://localhost", "X-Title": "Helix"},
                            model="openrouter/auto",
                            messages=messages
                        )
                        response = "🔎 *(Web searched)*\n\n" + completion2.choices[0].message.content

                thinking_placeholder.empty()

            st.session_state.chat_history.append({"role": "assistant", "content": response})
            save_memory(st.session_state.chat_history)
            placeholder = st.empty()
            type_text(response, placeholder)
        except Exception as e:
            st.error(f"SYSTEM ERROR: {str(e)}")
    st.rerun()
