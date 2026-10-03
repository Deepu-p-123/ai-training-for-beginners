
# ✍️ Handwritten Digit Classification with an ANN (Keras)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deepu-p-123/ai-training-for-beginners/blob/main/02-MNIST/Day2_Handwritten_Digit_ANN_Keras.ipynb)

> **Day 2 of AI Training for Beginners.** Teach a computer to read handwritten digits (0-9) using your first **Artificial Neural Network**. No installation needed, everything runs free in your browser.

👆 **Click the badge above to launch the notebook in Google Colab.**

---

## 📑 Table of Contents

1. [What is this project?](#-what-is-this-project)
2. [What you will learn](#-what-you-will-learn)
3. [Before you start](#-before-you-start)
4. [How to run the notebook](#-how-to-run-the-notebook)
5. [The big picture](#-the-big-picture)
6. [Step-by-step walkthrough](#-step-by-step-walkthrough)
7. [Understanding the model](#-understanding-the-model)
8. [Key terms explained](#-key-terms-explained)
9. [What results to expect](#-what-results-to-expect)
10. [Experiments to try](#-experiments-to-try)
11. [Troubleshooting & FAQ](#-troubleshooting--faq)
12. [Take-home challenges](#-take-home-challenges)
13. [Further learning](#-further-learning)

---

## 🎯 What is this project?

Every time you unlock a phone with your face, deposit a cheque through a banking app, or see a post office sort mail automatically, a computer is **recognizing patterns in images**. This project is the "Hello World" of that field.

You will build a **neural network** that looks at a tiny 28 × 28 picture of a handwritten digit and says which number it is. By the end, you will even test it on **your own handwriting**.

| | |
|---|---|
| **Dataset** | MNIST: 70,000 images of handwritten digits |
| **Task** | Image classification (10 classes: digits 0-9) |
| **Model** | ANN (Artificial Neural Network), also called a fully connected / dense network |
| **Library** | TensorFlow / Keras |
| **Platform** | Google Colab (free, runs in the browser) |
| **Difficulty** | Beginner |
| **Time needed** | About 2 hours including experiments (the training itself takes only a few minutes) |
| **Type of ML** | Supervised learning → classification |

---

## 🎓 What you will learn

By the end of this notebook you will be able to:

- ✅ Explain **how a computer "sees" an image** (as a grid of numbers)
- ✅ **Load and explore** an image dataset
- ✅ **Preprocess** data with normalization and one-hot encoding
- ✅ **Build** a neural network with Keras layer by layer
- ✅ Explain **epochs, batch size, loss, optimizer and activation functions**
- ✅ **Train** a model and read its learning curves
- ✅ Detect **overfitting** and know how to reduce it (dropout)
- ✅ **Evaluate** a model with accuracy, a confusion matrix, precision and recall
- ✅ **Predict** on your own handwritten digits
- ✅ **Experiment** like a real ML engineer by changing settings and comparing results

---

## 📋 Before you start

### What you need
- A **Google account** (for Google Colab)
- A **laptop or desktop** with a web browser (Chrome recommended)
- An internet connection

### What you do *not* need
- ❌ No software to install
- ❌ No powerful computer or graphics card (Colab provides one for free)
- ❌ No advanced maths

### Helpful background (but not required)
- Basic Python (variables, lists, loops, functions)
- The idea of **supervised learning** (features and labels) from Day 1

---

## 🚀 How to run the notebook

### Option 1: Google Colab (recommended)

1. Click the **Open in Colab** badge at the top of this page.
2. If Colab shows *"This notebook was not authored by Google"*, click **Run anyway**. The notebook is safe.
3. *(Optional but faster)* Switch to a free GPU: `Runtime → Change runtime type → T4 GPU → Save`. This model also trains fine on the normal CPU.
4. Run the cells **one at a time**: click a cell and press **`Shift + Enter`**.
5. **Read the explanation above each code cell before running it.** That is where the learning happens.

> 💡 **Tip:** Run the cells **in order from top to bottom.** Later cells depend on earlier ones. If you get a "name is not defined" error, you probably skipped a cell. Use `Runtime → Run all` to start fresh.

### Option 2: Upload to Colab manually

1. Download the file `Day2_Handwritten_Digit_ANN_Keras.ipynb` from this folder.
2. Go to [colab.research.google.com](https://colab.research.google.com).
3. Choose **File → Upload notebook** and select the file.

### Option 3: Run on your own computer (advanced)

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pillow jupyter
jupyter notebook Day2_Handwritten_Digit_ANN_Keras.ipynb
```

> ⚠️ Step 13 (uploading your own handwriting) uses `google.colab` and only works inside Colab. If running locally, skip it or load your image with `PIL.Image.open("your_file.png")`.

---

## 🗺️ The big picture

Here is the whole project in one picture:

```
 ┌───────────┐   ┌────────────┐   ┌──────────┐   ┌───────────┐   ┌────────────┐
 │ 1. LOAD   │ → │ 2. PREPARE │ → │ 3. BUILD │ → │ 4. TRAIN  │ → │ 5. EVALUATE│
 │ MNIST     │   │ normalize, │   │ the ANN  │   │ model.fit │   │ & PREDICT  │
 │ images    │   │ one-hot    │   │ (layers) │   │           │   │ new digits │
 └───────────┘   └────────────┘   └──────────┘   └───────────┘   └────────────┘
```

This is the same **ML pipeline** you learned on Day 1 (data → clean → model → train → evaluate), now applied to images with a neural network.

---

## 🔍 Step-by-step walkthrough

Here is what each step of the notebook does and **why**.

### Step 1: Import the libraries
We load the tools we need.

| Library | Purpose |
|---|---|
| `numpy` | Fast math on arrays. Images are arrays of numbers. |
| `matplotlib` | Draw images and graphs. |
| `seaborn` | Make the colourful confusion-matrix heatmap. |
| `tensorflow` / `keras` | Build and train the neural network. |
| `sklearn` | Evaluation tools (confusion matrix, classification report). |

### Step 2: Load the MNIST dataset
**MNIST** has **70,000** images of handwritten digits, split into:
- **60,000 training images**: the model learns from these.
- **10,000 test images**: the model never sees these while learning. They act as the final exam.

The data has four parts: `x_train`, `y_train`, `x_test`, `y_test`.
- **`x`** = the images (the *features*)
- **`y`** = the correct digit for each image (the *labels*)

Shapes: `x_train` is `(60000, 28, 28)` → 60,000 images, each 28 pixels tall and 28 pixels wide.

### Step 3: Look at the data
Always look at your data first! We display sample images and discover something important:

> 🧠 **A computer does not see a picture. It sees a grid of numbers.** Each pixel is a number from **0 (black)** to **255 (white)**. A 28 × 28 image is just 784 numbers.

We also draw a bar chart to check that the classes are **balanced** (about 6,000 images per digit). If one digit dominated, the model could cheat by guessing it all the time.

### Step 4: Preprocess the data
Two small but important changes:

**a) Normalization:** divide every pixel by 255 so values are between **0 and 1**.
*Why?* Small, consistent numbers make learning smooth and stable, like walking down a gentle slope instead of a cliff.

**b) One-hot encoding of labels:** turn the label `3` into `[0,0,0,1,0,0,0,0,0,0]`.
*Why?* The output layer has 10 neurons (one per digit), so the correct answer must also be 10 numbers: a `1` in the right position and `0` elsewhere.

| Label | One-hot vector |
|:---:|---|
| 0 | `[1,0,0,0,0,0,0,0,0,0]` |
| 3 | `[0,0,0,1,0,0,0,0,0,0]` |
| 7 | `[0,0,0,0,0,0,0,1,0,0]` |

### Step 5: Build the ANN
We stack layers like building blocks:

```
 Input image      Flatten        Dense (128)      Dense (64)       Dense (10)
  28 × 28    →   784 numbers →     ReLU      →      ReLU      →     Softmax
   (grid)         (a list)      hidden layer 1   hidden layer 2   output layer
```

See [Understanding the model](#-understanding-the-model) below for the details.

### Step 6: Compile the model
Before training we choose three things:

| Setting | Our choice | What it means |
|---|---|---|
| **Optimizer** | `adam` | The algorithm that adjusts the weights to reduce error. Imagine a hiker walking downhill in fog, feeling for the lowest point. |
| **Loss** | `categorical_crossentropy` | A score of **how wrong** the model is. Training tries to make it small. |
| **Metric** | `accuracy` | The percentage of correct predictions. This is what *we* watch. |

### Step 7: Train the model
`model.fit()` is where learning happens. We use `epochs=10`, `batch_size=128` and `validation_split=0.1`.

Each training step does four things:
1. **Forward pass:** images flow through the network → predictions.
2. **Loss calculation:** compare predictions with the true labels.
3. **Backpropagation:** work out how much each weight caused the error.
4. **Gradient descent:** nudge every weight a little to reduce the error.

This repeats thousands of times, and the network gets better and better.

### Step 8: Plot the learning curves
Two graphs (accuracy and loss) for training vs validation data. They tell you whether the model is learning well, overfitting or underfitting. See [What results to expect](#-what-results-to-expect).

### Step 9: Evaluate on the test set
The real final exam: 10,000 images the model has **never seen**. This gives an honest estimate of real-world performance.

### Step 10: Make predictions
`model.predict()` gives 10 **probabilities** for each image. The highest one is the model's answer, found with `np.argmax()`. We show predictions with a 🟢 green title (correct) or 🔴 red title (wrong), plus the model's confidence.

### Step 11: Confusion matrix and classification report
Accuracy is one number. The **confusion matrix** shows *which digits get mixed up*:
- **Rows** = the actual digit, **columns** = the predicted digit.
- The **diagonal** = correct predictions.
- Off-diagonal cells = mistakes (e.g. actual **4** predicted as **9**).

The **classification report** adds **precision**, **recall** and **F1-score** for every digit.

### Step 12: Look at the mistakes
We display images the model got wrong. Many are so messy that even humans struggle. Studying errors is the best way to improve a model.

### Step 13: Test on your own handwriting ✍️
Write a digit on paper, take a photo, crop it, and upload it. The notebook converts it to 28 × 28 grayscale, inverts the colours if needed (MNIST digits are *white on black*), and predicts.

**For best results:** use a **thick, dark pen** on **white paper**, write the digit **big and centered**, and crop close to the digit.

### Step 14: Experiments
A helper function `build_and_train()` lets you change neurons, layers, activation, dropout and learning rate. See [Experiments to try](#-experiments-to-try).

### Step 15: Save the model
Save your trained model to a file with `model.save()` and load it back later without retraining.

---

## 🧠 Understanding the model

### Layer by layer

| Layer | Code | What it does |
|---|---|---|
| **Flatten** | `Flatten(input_shape=(28, 28))` | Unrolls the 28 × 28 grid into a list of **784** numbers. Dense layers need a flat list. |
| **Hidden 1** | `Dense(128, activation="relu")` | 128 neurons. Each one looks at **all 784 pixels** and learns a different pattern. |
| **Hidden 2** | `Dense(64, activation="relu")` | 64 neurons that combine the first layer's patterns into higher-level ones. |
| **Output** | `Dense(10, activation="softmax")` | 10 neurons, one per digit. Softmax turns the scores into **probabilities that add up to 1**. |

### Number of parameters (the "knobs" the model learns)
For a Dense layer: **parameters = (inputs × neurons) + neurons (biases)**

| Layer | Calculation | Parameters |
|---|---|---:|
| Dense 1 | 784 × 128 + 128 | 100,480 |
| Dense 2 | 128 × 64 + 64 | 8,256 |
| Dense 3 (output) | 64 × 10 + 10 | 650 |
| **Total** | | **109,386** |

For comparison, ChatGPT-style models have *hundreds of billions* of parameters.

### Activation functions

| Function | Formula (idea) | Used for |
|---|---|---|
| **ReLU** | `max(0, x)`: negatives become 0, positives pass through | Hidden layers. It lets the network learn **non-linear** patterns. |
| **Softmax** | Converts scores to probabilities that sum to 1 | Output layer for multi-class problems |

> ❓ **Why do we need activation functions?** Without them, stacking many layers would still behave like *one* straight-line function, and the network could not learn complex patterns. Experiment C in the notebook proves this.

### What happens to one image

```
 Image of "7" → 784 pixel values (0 to 1)
              → Hidden layer 1 (128 neurons) → ReLU
              → Hidden layer 2 (64 neurons)  → ReLU
              → Output (10 scores) → Softmax
              → [0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.98, 0.00, 0.01]
                                                              ↑
                                                  Highest = digit 7 ✅
```

---

## 📖 Key terms explained

| Term | Simple meaning |
|---|---|
| **Pixel** | One tiny dot of an image, a number from 0 to 255 |
| **Neuron** | A tiny calculator: weighted sum of inputs + bias, then an activation function |
| **Weight** | How much a neuron cares about a particular input |
| **Bias** | A shift that lets a neuron adjust its output |
| **Layer** | A group of neurons working side by side |
| **Dense layer** | Every neuron connects to every neuron in the previous layer |
| **Activation function** | Adds non-linearity (ReLU, Softmax, Sigmoid) |
| **Epoch** | One complete pass through the entire training data |
| **Batch size** | How many images are processed before the weights are updated |
| **Loss function** | Measures how wrong the model is. Lower is better. |
| **Optimizer** | The method that updates the weights to reduce loss (Adam, SGD) |
| **Learning rate** | The size of each step the optimizer takes |
| **Backpropagation** | Calculating how each weight contributed to the error |
| **Gradient descent** | Moving the weights downhill to reduce the loss |
| **Overfitting** | Memorizing training data and failing on new data |
| **Underfitting** | The model is too simple or hasn't trained enough to learn the pattern |
| **Dropout** | Randomly switching off neurons during training to reduce overfitting |
| **Validation set** | Data held back during training to check progress |
| **Test set** | Unseen data used for the final evaluation |
| **Precision** | Of everything predicted as "7", how many really were 7? |
| **Recall** | Of all real 7s, how many did the model find? |
| **F1-score** | A balance of precision and recall |

---

## 📊 What results to expect

Your numbers will differ slightly from run to run, because the starting weights are random. That is normal.

| Measure | Typical result |
|---|---|
| Training accuracy after 10 epochs | about 99% |
| Validation accuracy | about 97-98% |
| **Test accuracy** | **about 97-98%** |
| Misclassified test images | roughly 200-300 out of 10,000 |

### How to read the learning curves

| What you see | What it means |
|---|---|
| ✅ Accuracy rises, loss falls, training and validation lines stay close | **Healthy learning** |
| ⚠️ Training accuracy keeps rising, but validation accuracy flattens or drops (validation loss goes up) | **Overfitting.** The model is memorizing. |
| ⚠️ Both accuracies stay low | **Underfitting.** The model is too small or needs more training. |

### Digits that are commonly confused
**4 ↔ 9**, **3 ↔ 5**, **7 ↔ 2** and **8 ↔ 3**. They look alike even to humans!

---

## 🧪 Experiments to try

The notebook includes a helper function so you can test ideas in one line. Fill in your own results:

| Experiment | What to change | What you will learn |
|---|---|---|
| **A: Network size** | Neurons: 8, 32, 128, 512 | Do bigger networks always do better? |
| **B: Depth** | Hidden layers: 0, 1, 2, 4 | Why hidden layers matter (0 layers = a very simple model) |
| **C: Activation** | `relu` vs `linear` | Why non-linearity is essential |
| **D: Learning rate** | 0.00001, 0.001, 0.1 | Too small = slow, too big = unstable |
| **E: Dropout** | 0.0 vs 0.3 | How dropout fights overfitting |

📝 Write down your results:

| Experiment | Setting | Test accuracy | What I noticed |
|---|---|---|---|
| A | 8 neurons | | |
| A | 512 neurons | | |
| B | 0 hidden layers | | |
| B | 4 hidden layers | | |
| C | linear activation | | |
| D | learning rate 0.1 | | |
| E | dropout 0.3 | | |

---

## 🛠️ Troubleshooting & FAQ

<details>
<summary><b>"NameError: name 'x_train' is not defined" (or similar)</b></summary>

You skipped a cell or ran them out of order. Use **Runtime → Run all**, or run the cells again from the top.
</details>

<details>
<summary><b>I see a warning about <code>input_shape</code> or <code>Do not pass an input_shape/input_dim argument to a layer</code></b></summary>

This is a harmless warning from newer Keras versions. The notebook still works correctly. You can ignore it.
</details>

<details>
<summary><b>My accuracy is different from the numbers in this README</b></summary>

That is completely normal. Neural networks start with random weights, so every run gives slightly different results (usually within about 1%).
</details>

<details>
<summary><b>Training is slow</b></summary>

Switch to a GPU: <code>Runtime → Change runtime type → T4 GPU</code>. If the GPU is unavailable, the CPU works too, and training this small model takes only a few minutes.
</details>

<details>
<summary><b>Colab disconnected, or I lost my variables</b></summary>

Colab sessions time out after inactivity. Reconnect and use <b>Runtime → Run all</b> to rebuild everything.
</details>

<details>
<summary><b>The model predicts my own handwriting wrongly</b></summary>

This is expected and a good lesson! The model was trained on clean, centered digits. Try: a thicker pen, writing bigger, centering the digit, cropping tightly, and using a plain white background. A model only performs well on data **similar to what it was trained on**.
</details>

<details>
<summary><b>The file upload in Step 13 does not work</b></summary>

The upload cell only works inside **Google Colab**. Also check that you uploaded an image file (<code>.jpg</code> or <code>.png</code>).
</details>

<details>
<summary><b>Why not use a CNN?</b></summary>

CNNs (Convolutional Neural Networks) are better for images, because they look at neighbouring pixels together. An ANN treats every pixel independently. We start with an ANN because it is simpler to understand. CNNs are the next topic! 🚀
</details>

---

## 🚀 Take-home challenges

**Easy**
- 🔹 Change the number of epochs to 3, then to 20. What happens to the curves?
- 🔹 Show 30 misclassified images instead of 15.
- 🔹 Test the model on 5 of your own digits. How many does it get right?

**Medium**
- 🔸 Repeat the whole project on **Fashion-MNIST** (`keras.datasets.fashion_mnist`). It has the same format but contains clothing items, so it is harder!
- 🔸 Add `EarlyStopping` so training stops when validation loss stops improving.
- 🔸 Try a different optimizer (`"sgd"`, `"rmsprop"`) and compare.

**Hard**
- 🔺 Reach **98.5% or higher** test accuracy using only an ANN.
- 🔺 Build a small web app that predicts digits (hint: try Gradio).

---

## 📚 Further learning

- 🎥 [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi): beautiful visual explanations
- 🧪 [TensorFlow Playground](https://playground.tensorflow.org): train a neural network visually in your browser
- 📘 [Keras documentation](https://keras.io): official guides
- 📗 [MNIST dataset page](http://yann.lecun.com/exdb/mnist/): the original dataset by Yann LeCun, Corinna Cortes and Christopher Burges

---

## ➡️ What's next?

Now that you understand ANNs, the next step is **Convolutional Neural Networks (CNNs)**, networks designed to understand images, and then **Transformers**, the technology behind ChatGPT.

---

⭐ **Found this helpful?** Star the repository and share it with a friend who is starting out in AI!
