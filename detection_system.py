import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import tkinter as tk
from tkinter import messagebox

# Load Dataset
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

fake_df["label"] = 0
true_df["label"] = 1

data = pd.concat([fake_df, true_df], axis=0)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# FASTEST MODEL → Linear SVM
model = LinearSVC()
model.fit(X_train, y_train)

# GUI Setup
root = tk.Tk()
root.title("Fake News Predictor")
root.geometry("700x550")
root.config(bg="#1a1a1a")

# Heading
heading = tk.Label(
    root,
    text="Fake News Predictor",
    font=("Arial", 22, "bold"),
    fg="#00eaff",
    bg="#1a1a1a")
heading.pack(pady=15)

# Textbox Label
label = tk.Label(
    root,
    text="Enter the news text below:",
    font=("Arial", 14),
    fg="white",
    bg="#1a1a1a")
label.pack()

# Textbox
textbox = tk.Text(
    root,
    height=15,
    width=80,
    font=("Arial", 12),
    bg="#2b2b2b",
    fg="white",
    wrap="word")
textbox.pack(pady=10)

# Result label
result_label = tk.Label(
    root,
    text="",
    font=("Arial", 18, "bold"),
    fg="yellow",
    bg="#1a1a1a")
result_label.pack(pady=15)

# Prediction Function
def predict_news():
    news = textbox.get("1.0", "end").strip()
    if news == "":
        messagebox.showwarning("Error", "Please enter some news text!")
        return

    vector = vectorizer.transform([news])
    prediction = model.predict(vector)[0]

    if prediction == 1
        result_label.config(text="✔ This news is REAL", fg="#00ff00")
    else:
        result_label.config(text="✘ This news is FAKE", fg="#ff3333")

# Predict button
predict_btn = tk.Button(
    root,
    text="Predict News",
    command=predict_news,
    font=("Arial", 14, "bold"),
    bg="#00eaff",
    fg="black",
    padx=15,
    pady=7)
predict_btn.pack(pady=10)

root.mainloop()


