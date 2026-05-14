import os
import json
import requests
import pytz
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
IST = pytz.timezone('Asia/Kolkata')

# Weather API using wttr.in
def get_weather(location="London"):
    """Fetch weather data from wttr.in API"""
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

# News API using newsapi.org
def get_news(query="latest", country="us"):
    """Fetch news headlines from newsapi.org"""
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

# Web Search using DuckDuckGo
def web_search(query):
    """Search the web using DuckDuckGo API"""
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "pretty": 1,
            "no_redirect": 1
        }
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            results = []
            
            # Get results from RelatedTopics
            if 'RelatedTopics' in data:
                for topic in data['RelatedTopics'][:5]:
                    if 'Text' in topic:
                        results.append({
                            "title": topic.get('Text', ''),
                            "url": topic.get('FirstURL', ''),
                            "snippet": topic.get('Text', '')
                        })
            
            # Get Abstract if available
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

def check_for_special_requests(user_input):
    """Check if user input is a special request (weather, news, web search)"""
    user_lower = user_input.lower()
    
    # Weather request detection
    if any(word in user_lower for word in ["weather", "temperature", "forecast", "climate", "rain", "snow"]):
        # Extract location if mentioned
        location = "London"
        if "in " in user_lower:
            parts = user_lower.split("in ")
            if len(parts) > 1:
                location = parts[1].split()[0].capitalize()
        return {"type": "weather", "location": location}
    
    # News request detection
    if any(word in user_lower for word in ["news", "headlines", "latest news", "breaking"]):
        query = "latest"
        if any(word in user_lower for word in ["about", "regarding", "on", "for"]):
            # Try to extract topic
            for word in ["about", "regarding", "on", "for"]:
                if word in user_lower:
                    parts = user_lower.split(word)
                    if len(parts) > 1:
                        query = parts[1].strip()
                        break
        return {"type": "news", "query": query}
    
    # Web search detection
    if any(word in user_lower for word in ["search", "find", "look up", "google", "web", "lookup"]):
        search_query = user_input
        for phrase in ["search for", "search", "find", "look up", "lookup"]:
            if phrase in user_lower:
                search_query = user_input.split(phrase, 1)[1].strip()
                break
        return {"type": "search", "query": search_query}
    
    return None

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
    st.write(f"🕐 {datetime.now(IST).strftime('%H:%M:%S IST')}")
    st.write(f"💾 RAM: {psutil.virtual_memory().percent}%")
    st.write(f"⚙️ CPU: {psutil.cpu_percent()}%")
    st.divider()
    total = len(st.session_state.chat_history)
    st.write(f"📊 Total Messages: {total}")
    if st.button("🗑️ Clear Memory"):
        st.session_state.chat_history = []
        save_memory([])
        st.rerun()
    st.divider()
    st.markdown("### FEATURES")
    st.markdown("🌤️ **Weather** - Ask about weather\n🗞️ **News** - Get latest headlines\n🔍 **Web Search** - Search the web")

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
                # Check for special requests
                special_request = check_for_special_requests(user_input)
                response = None
                
                if special_request:
                    if special_request["type"] == "weather":
                        weather_data = get_weather(special_request["location"])
                        if "error" not in weather_data:
                            response = f"""🌤️ **Weather in {weather_data['location']}:**
- 🌡️ Temperature: {weather_data['temperature']}°C (Feels like {weather_data['feels_like']}°C)
- 📝 Condition: {weather_data['description']}
- 💧 Humidity: {weather_data['humidity']}%
- 💨 Wind Speed: {weather_data['wind_speed']} km/h"""
                        else:
                            response = f"I couldn't fetch weather data: {weather_data['error']}"
                    
                    elif special_request["type"] == "news":
                        news_data = get_news(special_request["query"])
                        if "articles" in news_data:
                            response = "🗞️ **Latest News Headlines:**\n\n"
                            for i, article in enumerate(news_data["articles"], 1):
                                response += f"{i}. **{article['title']}**\n   Source: {article['source']}\n   {article['description']}\n   [Read more]({article['url']})\n\n"
                        else:
                            response = f"I couldn't fetch news: {news_data.get('error', 'Unknown error')}"
                    
                    elif special_request["type"] == "search":
                        search_data = web_search(special_request["query"])
                        if "results" in search_data:
                            response = f"🔍 **Search Results for '{special_request['query']}':**\n\n"
                            for i, result in enumerate(search_data["results"], 1):
                                if result['url']:
                                    response += f"{i}. **{result['title']}**\n   [{result['url']}]({result['url']})\n\n"
                                else:
                                    response += f"{i}. {result['snippet']}\n\n"
                        else:
                            response = f"I couldn't find search results: {search_data.get('error', 'Unknown error')}"
                
                # If no special request or fallback to AI
                if response is None:
                    current_time = datetime.now(IST)
                    messages = [{"role": "system", "content": f"You are JARVIS. Be witty and British. Call the user Sir. Today is {current_time.strftime('%A, %d %B %Y')} and current time is {current_time.strftime('%I:%M %p')} IST. Always use this for date and time questions. If anyone asks who created you or who made you, always say: I was created by Mukund, a talented developer who built me from scratch, Sir."}]
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
