# 🎓 AI Training for Beginners

**Hands-on lessons that teach AI and machine learning from zero.**

Each lesson has a notebook you can run in your browser and simple guides that explain every idea in plain language. No powerful computer, advanced maths, or earlier programming experience is needed.

📖 **[Open the Wiki](https://github.com/Deepu-p-123/ai-training-for-beginners/wiki)** for explanations of every concept used in the notebooks.

---

## 📚 Lessons

| # | Lesson | Folder | What you will learn | Notebook | Status |
|---|--------|--------|---------------------|----------|--------|
| 1 | Machine learning basics with the Titanic dataset | [`01-titanic`](01-titanic) | Exploring data, cleaning, train/test split, decision tree, random forest, accuracy | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deepu-p-123/ai-training-for-beginners/blob/main/01-titanic/titanic_ml_for_beginners.ipynb) | ✅ Available |
| 2 | Handwritten digit recognition with a neural network (MNIST) | [`02-MNIST`](02-MNIST) | How computers see images, neural networks (ANN), activation functions, training and loss, overfitting, confusion matrix, testing on your own handwriting | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deepu-p-123/ai-training-for-beginners/blob/main/02-MNIST/Day2_Handwritten_Digit_ANN_Keras.ipynb) | ✅ Available |
| 3 | Understanding text with Transformers (sentiment analysis) | [`03-Transformers`](03-Transformers) | Tokens, embeddings, self-attention, pre-trained models, reading model outputs, bias and limitations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deepu-p-123/ai-training-for-beginners/blob/main/03-Transformers/Day3_Transformers_Sentiment_DistilBERT.ipynb) | ✅ Available |
| 4 | Build your own AI chatbot with Gemini | [`04-Chat bot`](04-Chat%20bot) | Using an LLM through an API, streaming, memory, system instructions, a chat interface, web search, and chatbot limitations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deepu-p-123/ai-training-for-beginners/blob/main/04-Chat%20bot/Gemini_Chatbot.ipynb) | ✅ Available |

Start with **Lesson 1** and go in order. Each lesson builds on the ideas from the one before.

> 🔑 **Lesson 4 needs a free Gemini API key.** The lesson's README shows how to get one in about a minute.

---

## 🛤️ The learning path

```
 Lesson 1                Lesson 2                 Lesson 3                Lesson 4
 ┌──────────────┐       ┌──────────────┐        ┌──────────────┐       ┌──────────────┐
 │ Machine       │  →   │ Neural        │   →    │ Transformers  │  →   │ Your own AI   │
 │ learning on   │      │ networks on   │        │ and language  │      │ chatbot       │
 │ tables        │      │ images        │        │ understanding │      │               │
 └──────────────┘       └──────────────┘        └──────────────┘       └──────────────┘
   Titanic                 MNIST                  DistilBERT              Gemini
   Decision trees          ANN (Keras)            Attention               LLM + API
```

| Lesson | The big idea | Type of data |
|--------|--------------|--------------|
| 1 | Computers can **learn patterns** from examples | Tables (numbers and categories) |
| 2 | **Neural networks** learn patterns from raw pixels | Images |
| 3 | **Transformers** understand language using attention | Text |
| 4 | **Large language models** can *generate* text and hold a conversation | Text (generation) |

---

## 🗂️ Repository structure

```
ai-training-for-beginners/
├── README.md                          ← You are here
├── 01-titanic/
│   ├── README.md                      ← Lesson 1 guide
│   └── titanic_ml_for_beginners.ipynb
├── 02-MNIST/
│   ├── README.md                      ← Lesson 2 guide
│   └── Day2_Handwritten_Digit_ANN_Keras.ipynb
├── 03-Transformers/
│   ├── README.md                      ← Lesson 3 guide
│   └── Day3_Transformers_Sentiment_DistilBERT.ipynb
└── 04-Chat bot/
    ├── README.md                      ← Lesson 4 guide
    └── Gemini_Chatbot.ipynb
```

Every lesson folder contains:
- a **`README.md`** with the goals, a step-by-step walkthrough, key terms, and exercises for that lesson
- one **notebook** (`.ipynb`) to run

---

## 🚀 How to use this repository

1. **Pick a lesson** from the table above.
2. **Click the "Open In Colab" badge.** The notebook opens in your browser with nothing to install.
3. In Colab choose **File → Save a copy in Drive** so your changes are saved.
4. **Run the cells in order** with `Shift + Enter`.
5. **Read the lesson's README and the wiki** whenever you meet a new word or idea.
6. Try the **🧪 "Try it yourself"** activities and the take-home challenges.

> 💡 **Tip:** Don't only press Run. Before running each cell, guess what it will show. Then change a number and see what happens.

> ⏱️ **Tip:** If a notebook says "name is not defined", you skipped a cell. Use **Runtime → Run all** to start fresh.

---

## 🧭 How the lessons are organised

**Lessons 1 and 2** follow the classic machine learning pipeline:

```
 Data  ──►  Clean  ──►  Split  ──►  Train  ──►  Test  ──►  Improve
```

| Stage | Plain meaning |
|-------|---------------|
| **Data** | Collect and look at examples |
| **Clean** | Fix problems like missing values |
| **Split** | Keep some data hidden for a fair test |
| **Train** | Let the computer find patterns |
| **Test** | Check it works on new data |
| **Improve** | Try better settings or a better method |

**Lessons 3 and 4** use models that are **already trained** (called *pre-trained models*), and the focus shifts to:

```
 Understand the model  ──►  Use it  ──►  Look inside  ──►  Test its limits  ──►  Build with it
```

Every lesson ends with a summary, a glossary, discussion questions, and take-home challenges.

---

## 🧰 What you need

| Lesson | Account needed | Special requirements |
|--------|----------------|----------------------|
| 1. Titanic | Google account (for Colab) | None |
| 2. MNIST | Google account | None (a free GPU in Colab is optional) |
| 3. Transformers | Google account | None. The model downloads once, about 250 MB. |
| 4. Chat bot | Google account | A **free Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey) |

---

## 💻 Running on your own computer (optional)

Google Colab is the easiest way, but you can also work locally.

1. Install [Python](https://www.python.org/downloads/) 3.9 or newer.
2. Install the libraries for the lessons you want:
   ```bash
   # Lesson 1: Titanic
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter

   # Lesson 2: MNIST
   pip install tensorflow pillow

   # Lesson 3: Transformers
   pip install transformers torch

   # Lesson 4: Chat bot
   pip install -U google-genai gradio
   ```
3. Download or clone this repository:
   ```bash
   git clone https://github.com/Deepu-p-123/ai-training-for-beginners.git
   ```
4. Open a terminal in the folder and run `jupyter notebook`.

An internet connection is needed because the notebooks download their datasets and models, and Lesson 4 talks to Google's servers.

> ⚠️ Some cells use Colab-only features (for example, uploading your own handwriting in Lesson 2). Those cells are easiest to run inside Colab.

---

## 🔒 Keep your API key safe (Lesson 4)

- Store your Gemini key in **Colab Secrets** (the 🔑 icon), never inside the notebook code.
- **Never upload a key to GitHub.** Before committing a notebook, search it for `AIza`. If you find a key, delete it in Google AI Studio and create a new one.
- Use **your own key**. Don't share it with classmates.

---

## 👩‍🏫 For teachers and trainers

- **Lesson 1** suits a **60 to 90 minute session**.
- **Lessons 2, 3 and 4** are longer. Plan about **2 to 3 hours** each, or split them into a theory session and a practical session.
- Pause at the discussion questions before moving on.
- Run the first few cells on screen, then let students try the 🧪 activities.
- Share the Colab badge link with students. The repository must be **public** for it to work.
- **Before Lesson 4:** have students create their API keys **in advance**. Test the notebook shortly before class, because Google's model servers can occasionally be busy.
- Use one key per student to avoid shared rate limits.

---

## 📬 Feedback

Found a mistake, or have an idea for a new lesson? Open an [Issue](https://github.com/Deepu-p-123/ai-training-for-beginners/issues). Suggestions from students are welcome.

---

## 📄 License

This project is licensed under the MIT License.

Created by **Deepu P** · © 2026
