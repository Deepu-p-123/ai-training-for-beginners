# 💬 Build Your Own AI Chatbot with Gemini

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deepu-p-123/ai-training-for-beginners/blob/main/04-Chat%20bot/Gemini_Chatbot.ipynb)

> **Day 4 of AI Training for Beginners.** Build a chatbot that can answer **any question you ask**, with memory, a custom personality and a real chat window, using Google's **Gemini Flash** model. Free, runs in your browser, no installation.

👆 **Click the badge above to launch the notebook in Google Colab.**

---

## 📑 Table of Contents

1. [What is this project?](#-what-is-this-project)
2. [What you will learn](#-what-you-will-learn)
3. [Before you start: get your API key](#-before-you-start-get-your-api-key)
4. [How to run the notebook](#-how-to-run-the-notebook)
5. [How does a chatbot work?](#-how-does-a-chatbot-work)
6. [Step-by-step walkthrough](#-step-by-step-walkthrough)
7. [Key terms explained](#-key-terms-explained)
8. [What results to expect](#-what-results-to-expect)
9. [Experiments to try](#-experiments-to-try)
10. [Limitations of chatbots](#-limitations-of-chatbots)
11. [Troubleshooting & FAQ](#-troubleshooting--faq)
12. [Take-home challenges](#-take-home-challenges)
13. [Further learning](#-further-learning)

---

## 🎯 What is this project?

Tools like ChatGPT, Gemini and Claude feel like magic, but building a chatbot of your own takes only a few lines of Python. In this project you connect to a powerful AI model over the internet and build up a chatbot piece by piece:

1. A first AI answer in 3 lines of code
2. Answers that **stream** word by word
3. A chatbot with **memory**
4. A chatbot with a **personality**
5. A **text chat loop**
6. A real **chat window** (like ChatGPT) using Gradio
7. A bonus: a chatbot that **searches the web**

| | |
|---|---|
| **Model** | `gemini-3.5-flash` (you can change it with one line) |
| **Task** | Conversational question answering (text generation) |
| **Library** | `google-genai` (Google's official Python library) and `gradio` |
| **Platform** | Google Colab (free) |
| **You need** | A free Gemini API key |
| **Difficulty** | Beginner |
| **Time needed** | About 2 to 3 hours |
| **Type of AI** | Generative AI, a Large Language Model (LLM) built on a Transformer |

> 🔗 **Connection to Day 3:** Day 3's DistilBERT *understands* text. Gemini *generates* text. Both are Transformers.

---

## 🎓 What you will learn

By the end of this notebook you will be able to:

- ✅ Explain how a chatbot works: **your code + an API + a Large Language Model**
- ✅ Get and **safely store an API key** (Colab Secrets)
- ✅ Send a prompt to an LLM and read the response and **token usage**
- ✅ Use **streaming** for answers that appear as they are written
- ✅ Explain why LLMs have **no memory**, and how chat history gives them one
- ✅ Control behaviour with **system instructions** (prompt engineering)
- ✅ Build a **chat interface** with Gradio
- ✅ Use **Google Search grounding** for up-to-date answers
- ✅ Recognise **hallucination, bias and privacy risks**
- ✅ Handle real-world **API errors** (rate limits, server busy)

---

## 🔑 Before you start: get your API key

An **API key** is like a password that tells Google who is using their AI model. It is free to create.

### 1. Create the key
1. Open **https://aistudio.google.com/apikey**
2. Sign in with your Google account.
3. Click **Create API key** and copy it.

### 2. Store it safely in Colab Secrets 🔒
1. Open the notebook in Colab (use the badge above).
2. Click the **🔑 key icon** in the left sidebar (**Secrets**).
3. Click **Add new secret**.
4. **Name:** `GEMINI_API_KEY` (exactly this, capital letters)
5. **Value:** paste your key.
6. Turn on **Notebook access**.

> If you skip this, the notebook simply **asks you to paste the key** when you run the connect cell. The text stays hidden as you type.

### ⚠️ Key safety rules

| Do | Don't |
|---|---|
| ✅ Keep the key in Colab Secrets | ❌ Paste the key directly into code cells |
| ✅ Create a new key if you think it leaked | ❌ Post the key in chats, screenshots or on GitHub |
| ✅ Use your own key | ❌ Share your key with friends |

> 🛡️ **Before uploading a notebook to GitHub,** search it for `AIza` (the usual start of a Gemini key). If you find it, delete that key in Google AI Studio and make a new one.

### What about cost?
Gemini has a **free tier** with usage limits (requests per minute and per day). Limits and available models change over time, so check the current numbers in Google AI Studio. This notebook makes only a small number of requests.

---

## 🚀 How to run the notebook

### Option 1: Google Colab (recommended)

1. Click the **Open in Colab** badge at the top of this page.
2. If Colab shows *"This notebook was not authored by Google"*, click **Run anyway**.
3. Add your API key to Colab Secrets (see above).
4. Run the cells **one at a time**: click a cell and press **`Shift + Enter`**.
5. **Read the explanation above each cell before running it.**

> 💡 Run the cells **in order from top to bottom.** Later cells depend on earlier ones. If you see "name is not defined", use **Runtime → Run all** to start fresh.

### Option 2: Upload to Colab manually
1. Download `Gemini_Chatbot.ipynb` from this folder.
2. Go to [colab.research.google.com](https://colab.research.google.com).
3. Choose **File → Upload notebook**.

### Option 3: Run on your own computer (advanced)

```bash
pip install -U google-genai gradio jupyter
export GEMINI_API_KEY="your-key-here"     # Windows PowerShell: $env:GEMINI_API_KEY="your-key-here"
jupyter notebook Gemini_Chatbot.ipynb
```

> ⚠️ The key-loading cell reads from Colab Secrets and otherwise asks you to paste the key. When running locally, simply paste the key when asked.

---

## 💡 How does a chatbot work?

```
  You type a question
        │
        ▼
 ┌─────────────────┐     internet      ┌──────────────────────┐
 │ Your Colab code  │ ───────────────▶ │ Gemini model (Google) │
 │  (the "client")  │ ◀─────────────── │ runs on Google's GPUs │
 └─────────────────┘   generated text   └──────────────────────┘
        │
        ▼
  Answer shown to you
```

Four big ideas:

1. **The model is not on your computer.** It is far too big. Your code sends a request over the internet through an **API** and gets text back.
2. **An API key** identifies you to Google.
3. **LLMs generate text one token at a time**, always predicting the next token.
4. **LLMs have no memory.** Every request is independent. A chatbot "remembers" only because we **resend the whole conversation** each time.

---

## 🔍 Step-by-step walkthrough

### Step 1: Get your API key
Covered [above](#-before-you-start-get-your-api-key).

### Step 2: Install the library and connect
Installs `google-genai` (Google's official library) and `gradio` (for the chat window), and then connects.

The connect cell does four useful things:
- **Reads your key** from Colab Secrets, or asks you to paste it.
- **Cleans it up:** removes accidental spaces and handles unusual formats.
- **Retries automatically** (up to 5 times) if Google's servers are busy.
- **Sets the model** in one variable: `MODEL_NAME = "gemini-3.5-flash"`.

> 🔄 **Want a newer or different model?** Change only the `MODEL_NAME` line. A helper cell lists the Flash models available to your key.

### Step 3: Your first AI response 🚀
The simplest possible call:

```python
response = client.models.generate_content(
    model=MODEL_NAME,
    contents="Explain what a neural network is in 3 simple sentences."
)
print(response.text)
```

Then we look at **token usage**: how many tokens your prompt used, how many the answer used, and the total. Tokens are what AI services **count and charge for**, and they are the same idea you met in Day 3.

### Step 4: Streaming ⚡
Waiting for a long answer to finish feels slow. With `generate_content_stream()`, the answer arrives in small **chunks** that we print as they come, just like ChatGPT.

### Step 5: Memory 🧠
First we prove that a plain request has **no memory**:

```
You: My name is Arjun and I love cricket.   → Bot greets Arjun
You: What is my name?                       → Bot doesn't know! (new conversation)
```

Then we fix it with a **chat session** (`client.chats.create()`). The chat object stores the conversation and sends the full history with every new message. We print the stored history to show that **this list is the "memory"**.

What this teaches us:
- The model itself learned nothing. We **resend the history**.
- Longer conversations use **more tokens**.
- Models have a **context window** (a maximum amount of text they can consider).
- A new chat object means **no memory**.

### Step 6: Personality 🎭
A **system instruction** tells the model how to behave for the whole conversation (role, tone, rules). Users don't see it, but the model follows it. In the notebook we create a patient teacher that explains things simply and ends with a thought-provoking question.

> Same model, one paragraph of instructions, completely different behaviour. This is **prompt engineering**.

### Step 7: A chatbot loop 💻
A `while True` loop that asks for your input, sends it to the model, streams the answer and repeats. Type **`quit`**, **`exit`** or **`bye`** to stop. It uses `try/except`, so a network problem prints a message instead of crashing.

### Step 8: A real chat window with Gradio 🖥️
**Gradio** turns a Python function into a ChatGPT-style web interface. Gradio calls our `respond(message, history)` function each time you send a message. The function:
1. Converts Gradio's history into the format Gemini expects.
2. Streams Gemini's answer back piece by piece.
3. Catches errors and shows a friendly message.

The interface includes a title, a description and clickable example questions.

> ⚠️ While the cell runs, Gradio may print a **public link**. Anyone who opens it uses **your API key's quota**. Share it only for demos, and **stop the cell** (⏹) when finished.

### Bonus: Search the web 🌐
LLMs only know what they learned in training, so they can be out of date. **Grounding with Google Search** lets the model look things up first:

```python
search_tool = types.Tool(google_search=types.GoogleSearch())
config = types.GenerateContentConfig(tools=[search_tool])
```

Compare the answer with and without the tool for a question about recent news.

---

## 📖 Key terms explained

| Term | Simple meaning |
|---|---|
| **LLM** | Large Language Model: a huge Transformer that predicts the next token |
| **API** | A way for programs to talk to a service over the internet |
| **API key** | Your password for using the API |
| **Client** | The code on your side that sends requests |
| **Prompt** | The input or instruction you give the model |
| **Response** | The text the model sends back |
| **Token** | A piece of text (a word or part of a word). Usage is counted in tokens. |
| **System instruction** | Hidden rules that shape the bot's role, tone and behaviour |
| **Prompt engineering** | Writing instructions that get better, more reliable answers |
| **Streaming** | Receiving the answer in pieces as it is written |
| **Chat session** | An object that stores the conversation history |
| **Context window** | The maximum amount of text the model can consider at once |
| **Rate limit** | The maximum number of requests allowed in a period of time |
| **Retry / backoff** | Trying again after waiting, with longer waits each time |
| **Grounding** | Backing answers with real, current data (such as web search) |
| **Hallucination** | A confident but false answer |
| **Gradio** | A Python library that builds web interfaces quickly |
| **Multimodal** | A model that handles text, images, audio and more |

---

## 📊 What results to expect

LLM answers are different every time, so your output will not match anyone else's word for word. That is normal. Here is what you should see:

| Notebook step | What you should see |
|---|---|
| **Step 3** | A short, clear explanation, then token counts (prompt tokens fewer than answer tokens) |
| **Step 5, no memory** | The bot says it doesn't know your name |
| **Step 5, with chat** | The bot correctly says "Arjun" and "cricket" |
| **Step 6** | A short, simple answer ending in a question to the student |
| **Step 7** | A flowing conversation that remembers earlier messages |
| **Step 8** | A chat window where answers appear gradually |
| **Hallucination test** | Ideally "I'm not sure", but sometimes a confident, invented answer. That's the lesson. |

---

## 🧪 Experiments to try

| # | Experiment | What you will learn |
|---|---|---|
| 1 | Change the `system_instruction` to *"You are a pirate"* or *"Reply only in rhyming verse"* | How instructions change behaviour |
| 2 | Ask *"Explain gravity"* with the instruction *"Explain as if I am 5 years old"*, then *"Explain as a physics professor"* | Audience control |
| 3 | In the memory demo, send 10 messages, then print `chat.get_history()` | How history grows (and uses tokens) |
| 4 | Ask about a made-up book or event, such as *"Who won the 2087 World Cup?"* | Whether it admits ignorance or hallucinates |
| 5 | Ask about last week's news with and without Google Search grounding | Why grounding matters |
| 6 | Ask the bot to reply in your own language (Malayalam, Tamil, Hindi...) | Multilingual ability |
| 7 | Build a specialist bot: study buddy, recipe helper, interview coach | Practical prompt engineering |

📝 Record your findings:

| Experiment | What I tried | What happened | What I learned |
|---|---|---|---|
| 1 | | | |
| 4 | | | |
| 5 | | | |

---

## ⚠️ Limitations of chatbots

Every AI engineer must know where chatbots go wrong:

| Issue | What it means | What to do |
|---|---|---|
| **Hallucination** | Confidently states false things. It predicts likely text, not verified facts. | Double-check anything important |
| **Knowledge cutoff** | Only knows what it saw in training | Add web search (grounding) |
| **Bias** | Can reflect biases in its training data | Test for fairness, don't trust blindly |
| **Privacy** | Data you send goes to Google's servers | Never send passwords or private documents |
| **Rate limits** | Free tier allows only so many requests | Wait, retry, and avoid spamming |
| **Over-trust** | A fluent answer *sounds* correct even when wrong | Verify with trusted sources |

---

## 🛠️ Troubleshooting & FAQ

<details>
<summary><b>"503 UNAVAILABLE" or "This model is currently experiencing high demand"</b></summary>

This is <b>not a mistake in your code</b>. Google's servers are temporarily overloaded. The notebook already retries automatically. If it still fails:
<ol>
<li>Wait 30 to 60 seconds and run the cell again.</li>
<li>Run the model-list cell and change <code>MODEL_NAME</code> to a different <code>flash</code> model.</li>
</ol>
Real apps handle this the same way: retry, then fall back to another model.
</details>

<details>
<summary><b>"429" or "quota" or "RESOURCE_EXHAUSTED"</b></summary>

You hit a <b>rate limit</b> (too many requests in a short time, or your daily free limit). Wait a minute and try again. If many students share <b>one</b> key, you will hit the limit quickly, so use your own key.
</details>

<details>
<summary><b>"API key not valid" or "400 INVALID_ARGUMENT"</b></summary>

The key was copied incorrectly (missing characters, a space or a line break), or it was deleted. Create a fresh key at <a href="https://aistudio.google.com/apikey">aistudio.google.com/apikey</a> and add it again.
</details>

<details>
<summary><b>"AttributeError: 'dict' object has no attribute 'strip'"</b></summary>

An older version of the notebook caused this when the key was returned in an unexpected format. The current notebook handles it. Download the latest <code>Gemini_Chatbot.ipynb</code> from this folder.
</details>

<details>
<summary><b>"404 model not found"</b></summary>

Model names change over time. Run the <b>"Which models can my key use?"</b> cell, copy a current <code>flash</code> model name into <code>MODEL_NAME</code>, and run the connect cell again.
</details>

<details>
<summary><b>The notebook can't find my secret</b></summary>

Check that: (1) the name is exactly <code>GEMINI_API_KEY</code>, (2) <b>Notebook access</b> is switched on, and (3) you have not misspelled it. Otherwise, just paste the key when the notebook asks.
</details>

<details>
<summary><b>The chat loop (Step 7) has no input box</b></summary>

Run the cell and look at the <b>bottom of the output</b> for a text box labelled "You:". Type there and press Enter. To stop the loop, type <code>quit</code> or press the ⏹ stop button.
</details>

<details>
<summary><b>The Gradio chat window doesn't appear</b></summary>

Make sure the install cell ran without errors and that you ran the cell that defines <code>assistant_config</code> (Step 7) before Step 8. If it still fails, restart with <b>Runtime → Restart session</b> and use <b>Run all</b>.
</details>

<details>
<summary><b>Why does the bot give a different answer each time?</b></summary>

LLMs choose words with a bit of randomness, so identical questions give slightly different answers. That is normal and by design.
</details>

<details>
<summary><b>Is my conversation private?</b></summary>

Your messages are sent to Google's servers to generate answers. Don't type passwords, personal ID numbers or confidential documents. Check Google's current data policy for the Gemini API if you need details.
</details>

---

## 🚀 Take-home challenges

**Easy**
- 🔹 Change the system instruction to make a bot with your own personality.
- 🔹 Make the chat loop end with a friendly goodbye that includes your name.
- 🔹 Ask the bot 10 different questions and note which answers you think are wrong.

**Medium**
- 🔸 Build a **specialist bot**: study buddy, recipe helper, travel guide or interview coach.
- 🔸 Make the bot answer in **Malayalam, Tamil or Hindi** by changing only the system instruction.
- 🔸 Add your own `examples` and a nicer `title` to the Gradio interface.

**Hard**
- 🔺 Upload an **image or PDF** and ask questions about it. Gemini is **multimodal**, so look up `types.Part.from_bytes`.
- 🔺 Add an **automatic fallback model**: if `MODEL_NAME` fails, try a second model.
- 🔺 Build a chatbot that answers questions from **your own notes** (this idea is called RAG).

---

## 📚 Further learning

- 🧪 [Google AI Studio](https://aistudio.google.com): try Gemini in the browser and create API keys
- 📘 [Gemini API documentation](https://ai.google.dev/gemini-api/docs): official guides and model list
- 📗 [Gradio documentation](https://www.gradio.app/docs): build more interfaces
- 🎥 [3Blue1Brown: Transformers and LLMs](https://www.youtube.com/@3blue1brown): visual explanations of how LLMs work
- 📖 Prompt engineering guides: search for "prompt engineering guide" from any major AI provider

---

## 🎉 You made it!

Over four days you went from **classical ML** (Day 1), to **neural networks** (Day 2), to **Transformers** (Day 3), to **building your own AI chatbot** (Day 4). That is the same journey the whole field has taken.

---

⭐ **Found this helpful?** Star the repository and share it with a friend who is starting out in AI!
