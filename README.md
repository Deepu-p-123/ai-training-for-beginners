# 🎓 AI Training for Beginners

**Hands-on lessons that teach AI and machine learning from zero.**

Each lesson has a notebook you can run in your browser and simple wiki pages that explain every idea in plain language. No powerful computer, advanced maths, or earlier programming experience is needed.

📖 **[Open the Wiki](https://github.com/Deepu-p-123/ai-training-for-beginners/wiki)** for explanations of every concept used in the notebooks.

---

## 📚 Lessons

| # | Lesson | Folder | What you will learn | Notebook | Status |
|---|--------|--------|---------------------|----------|--------|
| 1 | Machine learning basics with the Titanic dataset | [`01-titanic`](01-titanic) | Exploring data, cleaning, train/test split, decision tree, random forest, accuracy | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deepu-p-123/ai-training-for-beginners/blob/main/01-titanic/titanic_ml_for_beginners.ipynb) | ✅ Available |
| 2 | Spam filter | `02-spam-filter` | Working with text data | n/a | 🔜 Coming soon |
| 3 | Crop disease detection | `03-crop-disease` | Working with images | n/a | 🔜 Coming soon |

Start with **Lesson 1**. Later lessons build on the ideas introduced there.

---

## 🗂️ Repository structure

```
ai-training-for-beginners/
├── README.md                  ← You are here
├── 01-titanic/
│   ├── README.md              ← Lesson 1 guide
│   └── titanic_ml_for_beginners.ipynb
├── 02-spam-filter/            ← Coming soon
└── 03-crop-disease/           ← Coming soon
```

Every lesson folder contains:
- a **`README.md`** with the goals, steps, and exercises for that lesson
- one or more **notebooks** (`.ipynb`) to run

---

## 🚀 How to use this repository

1. **Pick a lesson** from the table above.
2. **Click the "Open In Colab" badge.** The notebook opens in your browser with nothing to install.
3. In Colab choose **File → Save a copy in Drive** so your changes are saved.
4. **Run the cells in order** with `Shift + Enter`.
5. **Open the wiki** whenever you meet a new word or idea.
6. Try the **✏️ "Your turn"** activities and the challenge exercises.

> 💡 **Tip:** Don't only press Run. Before running each cell, guess what it will show. Then change a number and see what happens.

---

## 🧭 How every lesson is organised

All lessons follow the same pattern, so each one feels familiar:

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

---

## 💻 Running on your own computer (optional)

1. Install [Python](https://www.python.org/downloads/) 3.9 or newer.
2. Install the libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Download or clone this repository:
   ```bash
   git clone https://github.com/Deepu-p-123/ai-training-for-beginners.git
   ```
4. Open a terminal in the folder and run `jupyter notebook`.

An internet connection is needed because the notebooks download their datasets.

---

## 👩‍🏫 For teachers and trainers

- Each lesson suits a **60 to 90 minute session**.
- Pause at the 🤔 discussion questions before moving on.
- Run the first few cells on screen, then let students try the ✏️ activities.
- Share the Colab badge link with students. The repository must be **public** for it to work.

---

## 📬 Feedback

Found a mistake, or have an idea for a new lesson? Open an [Issue](https://github.com/Deepu-p-123/ai-training-for-beginners/issues). Suggestions from students are welcome.

---

## 📄 License

This project is licensed under the MIT License.

Created by **Deepu P** · © 2026
