
# 🤖 Transformers in Action: Sentiment Analysis with DistilBERT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deepu-p-123/ai-training-for-beginners/blob/main/03-Transformers/Day3_Transformers_Sentiment_DistilBERT.ipynb)

> **Day 3 of AI Training for Beginners.** Use a real **Transformer** (the technology behind ChatGPT) to read a sentence and decide if it is **positive 😊 or negative 😞**, and then open the model up to see *how* it works inside. Free, runs in your browser, no installation.

👆 **Click the badge above to launch the notebook in Google Colab.**

---

## 📑 Table of Contents

1. [What is this project?](#-what-is-this-project)
2. [What you will learn](#-what-you-will-learn)
3. [Before you start](#-before-you-start)
4. [How to run the notebook](#-how-to-run-the-notebook)
5. [Theory: what is a Transformer?](#-theory-what-is-a-transformer)
6. [Step-by-step walkthrough](#-step-by-step-walkthrough)
7. [Understanding the model](#-understanding-the-model)
8. [Key terms explained](#-key-terms-explained)
9. [What results to expect](#-what-results-to-expect)
10. [Experiments and challenges](#-experiments-and-challenges)
11. [Troubleshooting & FAQ](#-troubleshooting--faq)
12. [Take-home challenges](#-take-home-challenges)
13. [Further learning](#-further-learning)

---

## 🎯 What is this project?

When you use ChatGPT, Google Translate, or a search engine that understands your question, a **Transformer** is working behind the scenes. In this project you will use one to do **sentiment analysis**: deciding whether a piece of text expresses a positive or negative opinion. Companies use this to analyse customer reviews, social media comments and survey responses.

But we won't just *use* the model. We will **open the black box** and look at each stage: how text becomes numbers, how words "pay attention" to each other, and how the final decision is made.

| | |
|---|---|
| **Model** | `distilbert-base-uncased-finetuned-sst-2-english` |
| **Task** | Sentiment analysis (text classification: POSITIVE / NEGATIVE) |
| **Architecture** | Transformer (encoder), a small and fast version of Google's BERT |
| **Library** | 🤗 Hugging Face `transformers` + PyTorch |
| **Platform** | Google Colab (free, a normal CPU is enough, no GPU needed) |
| **Difficulty** | Beginner |
| **Time needed** | About 2 to 3 hours (the first model download takes a minute) |
| **Type of ML** | Supervised learning, using **transfer learning** (a pre-trained model) |

---

## 🎓 What you will learn

By the end of this notebook you will be able to:

- ✅ Explain **what a Transformer is** and why it replaced older models
- ✅ Use a **pre-trained model in 3 lines of code** with `pipeline`
- ✅ Explain **tokenization**: how text becomes numbers
- ✅ Explain **embeddings**: how numbers capture meaning
- ✅ Explain **self-attention** and **multi-head attention**, and **see them** in heatmaps
- ✅ Read a model's raw output (**logits**) and turn it into **probabilities** with softmax
- ✅ Find out **which words** drove a model's decision
- ✅ Analyse a set of real-world customer reviews
- ✅ Recognise **where Transformers fail**: sarcasm, neutral text, other languages and bias

---

## 📋 Before you start

### What you need
- A **Google account** (for Google Colab)
- A laptop or desktop with a web browser (Chrome recommended)
- An internet connection (the model is downloaded once, about **250 MB**)

### What you do *not* need
- ❌ No installation
- ❌ No GPU
- ❌ No advanced maths

### Helpful background (not required)
- Basic Python
- **Day 1** (supervised learning) and **Day 2** (neural networks, softmax). This notebook builds on both.

---

## 🚀 How to run the notebook

### Option 1: Google Colab (recommended)

1. Click the **Open in Colab** badge at the top of this page.
2. If Colab shows *"This notebook was not authored by Google"*, click **Run anyway**.
3. Run the cells **one at a time**: click a cell and press **`Shift + Enter`**.
4. **Read the explanation above each cell before running it.**

> 💡 Run the cells **in order from top to bottom.** Later cells depend on earlier ones. If you see "name is not defined", use **Runtime → Run all** to start fresh.

### Option 2: Upload to Colab manually
1. Download `Day3_Transformers_Sentiment_DistilBERT.ipynb` from this folder.
2. Go to [colab.research.google.com](https://colab.research.google.com).
3. Choose **File → Upload notebook**.

### Option 3: Run on your own computer (advanced)

```bash
pip install transformers torch pandas matplotlib seaborn jupyter
jupyter notebook Day3_Transformers_Sentiment_DistilBERT.ipynb
```

---

## 💡 Theory: what is a Transformer?

### The old problem
Older models (RNNs and LSTMs) read text **one word at a time, left to right**, like reading through a keyhole. By the end of a long sentence they had half-forgotten the beginning.

### The Transformer idea
Introduced in 2017 in the paper *"Attention Is All You Need"*. Instead of one word at a time, a Transformer looks at **all the words at once** and uses **self-attention** to decide *which words matter to which other words*.

> **Example:** *"The animal didn't cross the street because **it** was too tired."*
> What does **"it"** mean? Attention lets the model link **"it" → "animal"**.

### The Transformer family

| Model | Type | Good at | Example |
|---|---|---|---|
| **BERT / DistilBERT** | Encoder | *Understanding* text (classification, search) | **← this notebook** |
| **GPT** | Decoder | *Generating* text | ChatGPT |
| **T5 / BART** | Encoder + Decoder | Translation, summarization | |

### Pre-trained models and transfer learning
Training a Transformer from scratch needs huge data and computing power. Instead, someone else has already **pre-trained** the model on vast amounts of text, and then **fine-tuned** this one on movie reviews labelled positive or negative. We simply **reuse** it. This is called **transfer learning**.

---

## 🔍 Step-by-step walkthrough

### The pipeline we explore

```
 "I loved this movie!"
          │
          ▼
 ┌──────────────────┐
 │  1. TOKENIZER     │  text → pieces (tokens) → ID numbers
 └──────────────────┘
          ▼
 ┌──────────────────┐
 │  2. EMBEDDINGS    │  each ID → a list of 768 numbers (its "meaning")
 └──────────────────┘
          ▼
 ┌──────────────────┐
 │  3. TRANSFORMER   │  6 layers of SELF-ATTENTION: words talk to each
 │     LAYERS        │  other and update their meaning using context
 └──────────────────┘
          ▼
 ┌──────────────────┐
 │  4. CLASSIFIER    │  2 scores → softmax → NEGATIVE / POSITIVE
 └──────────────────┘
```

### Part 1: Theory
A short explanation of why Transformers exist, with the pipeline above and the model family table.

### Part 2: Setup
We install and import the libraries.

| Library | Purpose |
|---|---|
| `transformers` | Hugging Face's library with thousands of pre-trained models |
| `torch` (PyTorch) | The deep-learning engine that runs the model |
| `pandas` | Tables of data (for the review project) |
| `matplotlib`, `seaborn` | Charts and heatmaps |

### Part 3: Magic in 3 lines, the `pipeline`
Hugging Face's `pipeline` hides all the complexity. It downloads the model, tokenizes your text, runs the model and gives a readable answer:

```python
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("I absolutely loved this workshop!")
# [{'label': 'POSITIVE', 'score': 0.99...}]
```

- **label**: the model's decision
- **score**: its **confidence** from 0 to 1

You then test several sentences at once and type **your own sentence**.

### Part 4: Opening the black box 🔬
Here we load the **tokenizer** and the **model** separately and examine each stage.

**Model facts**

| Fact | Value |
|---|---|
| Total parameters | about **67 million** |
| Transformer layers | **6** |
| Attention heads per layer | **12** |
| Embedding size | **768** numbers per token |
| Output labels | `0 = NEGATIVE`, `1 = POSITIVE` |
| Maximum input length | **512 tokens** |

*(Compare with the Day 2 ANN, which had about 109 thousand parameters.)*

**Step 1️⃣ Tokenization: text → numbers**
Computers only understand numbers. The tokenizer splits text into **tokens** (words or word-pieces) and maps each to an **ID**.

```
"Transformers are absolutely amazing!"
   → ['transformers', 'are', 'absolutely', 'amazing', '!']
   → [19081, 2024, 7078, 6429, 999]    (IDs shown for illustration)
```
What to notice:
- Text is **lowercased** (the model is "uncased").
- Rare words are split into **sub-word pieces**. Pieces starting with `##` continue the previous piece, so the model never meets a truly "unknown" word.
- Punctuation gets its own token.

**Special tokens**

| Token | Meaning |
|---|---|
| `[CLS]` | Added at the **start**. Acts as a *summary* token. After attention, its vector represents the whole sentence, and **the classifier reads only this one**. |
| `[SEP]` | Added at the **end** to mark the sentence boundary. |
| `attention_mask` | `1` for real tokens, `0` for padding (used when batching sentences of different lengths). |

**Step 2️⃣ Embeddings: IDs → meaning vectors**
Each token ID becomes a vector of **768 numbers**. Words with similar meanings get similar vectors. We measure this with **cosine similarity** (1 = same direction, 0 = unrelated). For example, `good ↔ great` should score higher than `good ↔ bad` or `king ↔ banana`.

> ⚠️ These are the *raw input* embeddings, before attention adds context. Treat them as a rough guide.

**Steps 3️⃣ and 4️⃣: Transformer layers and classification**
We run the full model. It returns **logits** (raw scores). **Softmax** converts them into probabilities that add up to 1, the same softmax you used in Day 2.

We also print the **shape of the data at each layer**: `(1, tokens, 768)`. The shape never changes, but the numbers inside do, because each layer rewrites every token's vector using context from the other tokens.

### Part 5: Seeing attention 👀
Self-attention is the heart of the Transformer. For each token the model works out how much it should "look at" every other token.

- **Row** = the token that is *looking*
- **Column** = the token being *looked at*
- **Brighter cell** = more attention
- **Each row adds up to 1**

For the sentence *"The movie was not good at all"*, check whether **`good`** pays attention to **`not`**. That link is how the model understands that *"not good"* is negative.

You also see **all 12 attention heads** of the last layer side by side. Each head learns a *different* pattern, like 12 readers looking at the sentence from different angles. This is called **multi-head attention**.

### Part 6: Which words drive the decision? 🔎
Attention shows where the model *looks*, but what we really want is: **which words made it say POSITIVE or NEGATIVE?**

A simple trick: **remove one word at a time** and see how the "positive" probability changes.

- 🟢 Removing the word **lowers** the positive score → that word was pushing toward **positive**
- 🔴 Removing the word **raises** it → that word was pushing toward **negative**

Compare `"I love this phone"` with `"I do not love this phone"` to see how a single word flips the result.

### Part 7: Mini project, analyze customer reviews 🛍️
A real-world use case. We classify 12 product reviews, store the results in a table (`pandas` DataFrame), draw a chart of positive vs negative counts, and list the **least confident predictions**, the ones a human should double-check. You are encouraged to replace the reviews with your own data (canteen reviews, YouTube comments, tweets).

### Part 8: Limitations, where Transformers fail ⚠️
We test tricky sentences and run a simple bias check.

| Issue | Why it happens |
|---|---|
| **Sarcasm** | "Oh great, another Monday meeting." The words look positive, but the meaning is negative. |
| **Neutral text** | This model has only two classes, so it is forced to choose POSITIVE or NEGATIVE even for neutral facts. |
| **Other languages** | It was trained on **English** movie reviews only. |
| **Bias** | We change only a subject ("He", "She", "A rich man", "A poor man") in an otherwise identical sentence. Any difference in score reveals bias learned from the training data. |

### Part 9: Challenge time 🧪
Open-ended tasks to try your own ideas. See [Experiments and challenges](#-experiments-and-challenges).

---

## 🧠 Understanding the model

### What happens to one sentence

```
 "The movie was not good at all"
   │
   ▼ Tokenizer
 [CLS] the movie was not good at all [SEP]        ← tokens
 [101, 1996, 3185, 2001, 2025, 2204, 2012, 2035, 102]   ← IDs (illustrative)
   │
   ▼ Embeddings (768 numbers per token)
   │
   ▼ 6 Transformer layers (each with 12 attention heads)
   │   every token updates its meaning using all the other tokens
   │
   ▼ Take the [CLS] vector (summary of the whole sentence)
   │
   ▼ Classifier → 2 logits → Softmax
   │
 [NEGATIVE: 0.99,  POSITIVE: 0.01]   → 🎯 NEGATIVE
```

### Why attention helps with "not good"
A model that just counts positive and negative words sees "good" (positive) and gets confused. A Transformer lets the token **"good"** *attend to* **"not"**, so the meaning of "good" is **changed by its context** before the decision is made.

### Logits vs probabilities

| | Logits | Probabilities |
|---|---|---|
| **What** | Raw scores, can be any number (positive or negative) | Values between 0 and 1 that add up to 1 |
| **How** | Output of the final layer | Logits passed through **softmax** |
| **Example** | `[-3.2, 3.5]` | `[0.001, 0.999]` |

---

## 📖 Key terms explained

| Term | Simple meaning |
|---|---|
| **NLP** | Natural Language Processing: teaching computers to work with human language |
| **Sentiment analysis** | Deciding whether text expresses a positive or negative opinion |
| **Token** | A piece of text (a word or part of a word) |
| **Tokenizer** | Converts text ↔ token IDs |
| **Vocabulary** | The fixed list of all tokens the model knows |
| **Embedding** | A list of numbers representing a token's meaning |
| **Self-attention** | Each token decides which other tokens matter to it |
| **Attention head** | One of several parallel "readers" inside a layer |
| **Multi-head attention** | Several heads looking at the sentence from different angles at once |
| **Encoder** | The part of a Transformer that *understands* input (BERT is encoder-only) |
| **`[CLS]` token** | Special start token whose final vector summarizes the sentence |
| **Logits** | Raw output scores before softmax |
| **Softmax** | Turns scores into probabilities that sum to 1 |
| **Confidence** | The probability the model gives to its chosen label |
| **Pre-trained model** | A model already trained on massive data |
| **Fine-tuning** | Training a pre-trained model a little more on your own task |
| **Transfer learning** | Re-using knowledge from one task for another |
| **Hallucination / bias** | Wrong or unfair outputs learned from imperfect data |

---

## 📊 What results to expect

Exact numbers can vary slightly between library versions, but you should see patterns like these:

| Sentence type | Typical behaviour |
|---|---|
| Clearly positive ("I absolutely loved this workshop!") | **POSITIVE**, very high confidence (above 99%) |
| Clearly negative ("This was a complete waste of time.") | **NEGATIVE**, very high confidence |
| Mixed or mild ("The food was okay, nothing special.") | Lower confidence, since the model must still pick one side |
| Sarcasm ("Oh great, another Monday morning meeting.") | Often **wrong**, because the model reads the words, not the tone |
| Non-English text | Unreliable, because the model was trained on English |
| "not bad at all" / "not good at all" | Usually handled correctly thanks to attention |

### In the review mini-project
Most of the 12 reviews are classified correctly. The reviews where the model is **least confident** (mild or neutral wording) are the ones to check manually.

### Reading the attention heatmaps

| What you see | What it means |
|---|---|
| A bright cell at (`good`, `not`) | The word "good" is paying attention to "not" |
| Early layers: spread-out attention | The model is gathering broad context |
| Later layers: more focused attention | The model is zooming in on what matters for the decision |
| 12 heads look different | Each head specialises in a different kind of relationship |

---

## 🧪 Experiments and challenges

Try these inside the notebook:

1. **Fool the model.** Write a sentence that a human finds clearly positive, but the model calls NEGATIVE (or the reverse). Share it with the class!
2. **Intensity test.** Compare `"good"`, `"very good"`, `"extremely good"` and `"the best thing ever"`. Does the confidence increase?
3. **Order test.** Compare `"Good food but bad service"` with `"Bad service but good food"`. Does word order matter?
4. **Length test.** Paste a very long paragraph (more than 500 words). What happens? *(Hint: the 512-token limit.)*
5. **Pronoun test.** Visualize attention for *"The dog chased the cat because it was hungry"*. What does **"it"** attend to?
6. **Your own data.** Replace the review list with reviews of your college canteen, comments from a YouTube video, or tweets about a film.

📝 Record your findings:

| Experiment | Sentence I tried | Model said | Was it right? | What I learned |
|---|---|---|---|---|
| Fool the model | | | | |
| Intensity | | | | |
| Order | | | | |

---

## 🛠️ Troubleshooting & FAQ

<details>
<summary><b>The first run is slow, or it seems stuck at "Downloading"</b></summary>

The model (about 250 MB) is downloaded the first time. Wait a minute. After that it is cached for the session. You may see a warning about the `HF_TOKEN`. You can ignore it. You do not need a Hugging Face account for this notebook.
</details>

<details>
<summary><b>Error mentioning <code>attn_implementation</code> or "output_attentions"</b></summary>

This usually means the `transformers` library is old. Run <code>!pip install -U transformers</code> in a new cell, then use <b>Runtime → Restart session</b> and run all cells again.
</details>

<details>
<summary><b>"NameError: name 'model' / 'tokenizer' / 'classifier' is not defined"</b></summary>

You skipped a cell or ran them out of order. Use <b>Runtime → Run all</b>, or re-run the cells from the top.
</details>

<details>
<summary><b>The attention heatmap looks different from my friend's</b></summary>

That is fine. Heatmaps depend on the sentence, the layer you pick and the library version. The patterns, not the exact numbers, are what matter.
</details>

<details>
<summary><b>The model gives a wrong answer for my sentence</b></summary>

That is normal and a good lesson. See <a href="#-what-results-to-expect">What results to expect</a>. No model is perfect. Sarcasm, neutral text and other languages are hard.
</details>

<details>
<summary><b>Do I need a GPU?</b></summary>

No. DistilBERT is small and fast, so a normal Colab CPU is enough for every cell in this notebook.
</details>

<details>
<summary><b>Colab disconnected, or I lost my variables</b></summary>

Colab sessions time out after inactivity. Reconnect and use <b>Runtime → Run all</b> to rebuild everything.
</details>

<details>
<summary><b>What is the difference between DistilBERT and ChatGPT?</b></summary>

DistilBERT is an <b>encoder</b> that <i>understands</i> text (for example classifying sentiment). ChatGPT is a <b>decoder</b>-style model that <i>generates</i> text one token at a time. Both are Transformers and both use attention. DistilBERT is also far smaller: about 67 million parameters compared with hundreds of billions.
</details>

<details>
<summary><b>Why "Distil"BERT?</b></summary>

It is a <b>distilled</b> (compressed) version of BERT: smaller and faster, while keeping most of BERT's accuracy. That is why it runs comfortably on a free CPU.
</details>

---

## 🚀 Take-home challenges

**Easy**
- 🔹 Classify 10 sentences about your favourite film or food.
- 🔹 Find 3 sentences where the model is **less than 70% confident**.
- 🔹 Make the word-importance chart for 3 different sentences.

**Medium**
- 🔸 Try a different model for the same task: `"cardiffnlp/twitter-roberta-base-sentiment-latest"`. It adds a **neutral** class!
- 🔸 Try other `pipeline` tasks: `"summarization"`, `"translation_en_to_fr"`, `"text-generation"`, `"zero-shot-classification"`.
- 🔸 Collect 20 real reviews from a website and build your own sentiment summary.

**Hard**
- 🔺 **Fine-tune** DistilBERT on your own labelled dataset.
- 🔺 Build a small web app where a user types a sentence and sees the sentiment (hint: Gradio).

---

## 📚 Further learning

- 🎥 [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi): includes a visual explanation of attention and Transformers
- 📘 [Hugging Face LLM Course](https://huggingface.co/learn): free, hands-on lessons
- 📗 [Hugging Face Model Hub](https://huggingface.co/models): browse thousands of pre-trained models
- 📄 *"Attention Is All You Need"* (Vaswani et al., 2017): the original Transformer paper
- 🖼️ *"The Illustrated Transformer"* by Jay Alammar: a famous visual guide

---

## ➡️ What's next?

You have now seen **classical ML** (Day 1), **neural networks** (Day 2) and **Transformers** (Day 3). Next, you can use a large Transformer to build your **own AI chatbot** with Gemini.

---

⭐ **Found this helpful?** Star the repository and share it with a friend who is starting out in AI!
