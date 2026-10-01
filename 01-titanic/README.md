# 🚢 Lesson 1: Machine Learning Basics with the Titanic Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deepu-p-123/ai-training-for-beginners/blob/main/01-titanic/titanic_ml_for_beginners.ipynb)

In this lesson you will teach a computer to predict **whether a Titanic passenger survived**, using only details like age, sex, and ticket class. Along the way you will learn how machine learning works from start to finish.

---

## 🎯 What you will learn

- What machine learning is, and how it differs from normal programming
- How to explore a dataset with charts
- How to clean messy data (missing values, words to numbers)
- Why we split data into a **training set** and a **test set**
- How a **decision tree** makes predictions
- What **overfitting** is, and how to spot it
- How a **random forest** improves on a single tree
- How to measure results with **accuracy** and a **confusion matrix**
- Why AI can be unfair when it learns from biased data

## ⏱️ Time needed

About **60 to 90 minutes**.

## ✅ Before you start

- A Google account (for Colab) or Python installed on your computer
- An internet connection (the notebook downloads the dataset)
- No earlier programming or maths knowledge needed

---

## 🚀 How to start

1. Click the **Open in Colab** button at the top of this page.
2. In Colab choose **File → Save a copy in Drive** so your changes are saved.
3. Run each cell with **Shift + Enter**, in order from top to bottom.
4. Read the matching wiki page whenever you meet a new idea (links below).

---

## 📁 Files in this folder

| File | Purpose |
|------|---------|
| `titanic_ml_for_beginners.ipynb` | The lesson notebook |
| `README.md` | This page |

---

## 🗺️ The notebook at a glance

| Step | What happens |
|------|--------------|
| 0 | Set up the tools |
| 1 | Look at the data and understand each column |
| 2 | Explore with charts |
| 3 | Clean the data |
| 4 | Split into training and test sets |
| 5 | Train a decision tree |
| 6 | Test the model |
| 7 | See overfitting in action |
| 8 | Improve with a random forest |
| 9 | Predict for a passenger you create |
| 10 | Think about limits and fairness |

**What result to expect:** the decision tree scores roughly **80%** accuracy on the test data and the random forest slightly higher, around **80 to 82%**. Your numbers may differ a little.

---

## 📖 Learn more in the wiki

| Topic | Page |
|-------|------|
| What scikit-learn is, its datasets, and the Titanic data | [Scikit-Learn and Datasets](https://github.com/Deepu-p-123/ai-training-for-beginners/wiki/L1-Titanic-Scikit-Learn-and-Datasets) |
| Fixing missing values and encoding with pandas | [Data Cleaning with Pandas](https://github.com/Deepu-p-123/ai-training-for-beginners/wiki/L1-Titanic-Data-Cleaning-with-Pandas) |
| Decision tree, gini, overfitting, random forest | [Algorithms Used](https://github.com/Deepu-p-123/ai-training-for-beginners/wiki/L1-Titanic-Algorithms-Used) |

---

## ✏️ Challenge exercises

Try these after finishing the notebook:

1. Add a new feature `family_size = sibsp + parch + 1`. Does accuracy improve?
2. Try `LogisticRegression` instead of a decision tree. How does it compare?
3. Change `random_state` in the train/test split. Does accuracy change? Why?
4. Fill missing ages with the **mean** instead of the **median**. Does it matter?

---

## 🛠️ Common problems

| Problem | What to try |
|---------|-------------|
| `NameError: name 'df' is not defined` | A cell was skipped. Choose **Runtime → Restart and run all** |
| Dataset fails to load | Check your internet connection and run the cell again |
| Charts do not appear | Make sure the cell finished running (no spinning icon) |

---

⬅️ [Back to the main page](../README.md) · ➡️ Next lesson: *coming soon*
