# 🤖 JARVIS - AI Assistant with Memory

A sophisticated AI assistant with persistent memory, available in both **KivyMD mobile** and **Streamlit web** versions. JARVIS features a witty British personality and remembers all conversations.

---

## ✨ Features

- 🧠 **Persistent Memory** - Saves all conversations locally in JSON format
- 🎭 **JARVIS Personality** - British wit, sophisticated responses, calls you "Sir"
- 🌐 **Multi-Platform** - Available as mobile app (KivyMD) and web app (Streamlit)
- 🎨 **Dark Theme** - Sleek dark interface with cyan accents
- ⚡ **Fast Responses** - Powered by OpenRouter's free and paid models
- 📊 **Chat Statistics** - Track conversations and message counts
- 💾 **Export Capability** - Download chat history as JSON
- 🔐 **Secure API Keys** - Environment variable based configuration

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenRouter API key (free tier available at https://openrouter.ai)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mukundsingh820-lab/Jarvis.git
   cd Jarvis
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenRouter API key
   ```

---

## 🌐 Run Streamlit Web App

The Streamlit version is perfect for web browsers and laptops:

```bash
streamlit run jarvis_streamlit.py
```

**Features:**
- 🎨 Dark theme with cyan colors
- 💬 Real-time chat interface
- 📊 Chat statistics sidebar
- 💾 Export chat history
- 🗑️ Clear memory button
- ⏱️ Live system vitals (time, battery, CPU, memory)

**Access:** Open `http://localhost:8501` in your browser

---

## 📱 Run KivyMD Mobile App

The KivyMD version is for desktop/mobile:

```bash
python jarvis.py
```

**Features:**
- 📲 Touch-friendly interface
- ✨ Pulsing header animation
- ⌨️ Typewriter text effect
- 🔋 Battery indicator
- ⏰ Real-time clock
- 📜 Scrollable chat history

---

## 🔐 Security

Your API key is protected using environment variables:

1. API key is stored in `.env` (never committed to git)
2. `.gitignore` prevents accidental commits
3. Use `.env.example` as a template
4. Never share your `.env` file

**Setup:**
```bash
# .env.example
OPENROUTER_API_KEY=sk-or-v1-your-api-key-here

# .env (local only)
OPENROUTER_API_KEY=sk-or-v1-[your-actual-key]
```

---

## 📁 Project Structure

```
Jarvis/
├── jarvis.py                # KivyMD mobile app
├── jarvis_streamlit.py      # Streamlit web app
├── requirements.txt         # Python dependencies
├── .env.example            # API key template (safe to commit)
├── .env                    # API key (local only, in .gitignore)
├── .gitignore              # Prevents .env and memory from committing
├── jarvis_memory.json      # Chat history (auto-generated)
└── README.md               # This file
```

---

## 💾 Chat History

All conversations are automatically saved to `jarvis_memory.json`:

```json
[
  {
    "role": "user",
    "content": "Hello JARVIS"
  },
  {
    "role": "assistant",
    "content": "Good evening, Sir. How may I be of assistance?"
  }
]
```

**In Streamlit:**
- View in sidebar under "STATISTICS"
- Export as JSON using the export button
- Clear using the "Clear Memory" button

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | ✅ Yes |

### System Requirements

**KivyMD Version:**
- Display: 1080x1920px (optimized for mobile)
- RAM: 512MB+
- Python 3.8+

**Streamlit Version:**
- Any modern browser
- RAM: 256MB+
- Python 3.8+

---

## 🎭 JARVIS Personality

JARVIS is designed to be:
- 🇬🇧 Witty and British
- 🧠 Intelligent and articulate
- 👔 Sophisticated and formal (calls user "Sir")
- 💭 Contextual (remembers previous conversations)
- ⚡ Responsive and helpful

---

## 🐛 Troubleshooting

### API Key Not Found
```
❌ OPENROUTER_API_KEY not found in .env file!
```
**Solution:** Create `.env` file from `.env.example` and add your API key

### Connection Error
```
Connection refused to https://openrouter.ai/api/v1
```
**Solution:** Check internet connection and API key validity

### Memory File Not Found
```
jarvis_memory.json not found
```
**Solution:** First run will create it automatically

### Streamlit Not Starting
```
streamlit: command not found
```
**Solution:** Reinstall requirements: `pip install -r requirements.txt`

---

## 🔄 API Information

**Service:** OpenRouter  
**Base URL:** `https://openrouter.ai/api/v1`  
**Model:** `openrouter/free` (free tier available)  
**Pricing:** Free with rate limits, or paid for higher limits

Get your free API key: https://openrouter.ai/api/keys

---

## 📝 Example Conversations

**User:** "What's the weather like?"
```
JARVIS: I'm afraid I don't have access to real-time weather data, Sir. 
However, I suggest checking your local weather service or asking your device's 
built-in weather application. Is there something else I might assist you with?
```

**User:** "Tell me a joke"
```
JARVIS: Ah, very well, Sir. Why did the AI go to school? 
Because it wanted to improve its learning algorithms! 
I do apologize for the quality - my humor module requires calibration.
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Improve documentation
- Submit pull requests

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👨‍💻 Author

**mukundsingh820-lab**

---

## 📞 Support

For issues, questions, or suggestions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review existing GitHub issues
3. Create a new issue with detailed information

---

## 🎯 Future Enhancements

- [ ] Voice input/output support
- [ ] Multiple AI model selection
- [ ] Conversation summarization
- [ ] Theme customization
- [ ] Database integration for larger memory
- [ ] Multi-user support
- [ ] Mobile app distribution

---

**Made with ❤️ by mukundsingh820-lab**
