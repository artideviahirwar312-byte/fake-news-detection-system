import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import joblib
import requests
import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================
# LOAD TRAINED MODELS
# ============================
vectorizer = joblib.load("vectorizer.pkl")
ml_model = joblib.load("ml_model.pkl")
tokenizer = joblib.load("tokenizer.pkl")
dl_model = load_model("dl_model.h5")


# ============================
# HYBRID PREDICTION FUNCTION
# ============================
def hybrid_predict(text):
    # ML Model output
    ml_vec = vectorizer.transform([text])
    ml_cls = ml_model.predict(ml_vec)[0]
    ml_prob = 0.9 if ml_cls == 1 else 0.1

    # Deep Learning Model output
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=300)
    dl_prob = float(dl_model.predict(pad)[0][0])

    # Final average
    return (ml_prob + dl_prob) / 2


# ============================
# LIVE NEWS FETCHING
# ============================
NEWS_API_KEY = "9ac7eec77bbb466ca77d74ebf293bce2"

def fetch_latest_news():
    url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={NEWS_API_KEY}"
    response = requests.get(url).json()

    if response.get("status") != "ok":
        return None

    article = response["articles"][0]
    title = article.get("title", "")
    desc = article.get("description", "")
    return f"{title}\n\n{desc}"


# ============================
# GUI
# ============================
root = tk.Tk()
root.title("Fake News Detector")
root.geometry("800x600")
root.config(bg="#1a1a1a")

title = tk.Label(
    root, text="Real-Time Fake News Detector",
    font=("Arial", 25, "bold"), fg="#00eaff", bg="#1a1a1a"
)
title.pack(pady=10)

textbox = tk.Text(root, height=15, width=90, font=("Arial", 12), bg="#2b2b2b", fg="white")
textbox.pack(pady=15)

result_label = tk.Label(root, text="", font=("Arial", 18, "bold"), bg="#1a1a1a")
result_label.pack(pady=10)


def run_prediction():
    news = textbox.get("1.0", "end").strip()
    if news == "":
        messagebox.showerror("Error", "Please enter news text!")
        return

    prob = hybrid_predict(news)

    if prob >= 0.5:
        result_label.config(text="✔ REAL NEWS", fg="#00ff00")
    else:
        result_label.config(text="✘ FAKE NEWS", fg="#ff3333")


def fetch_and_predict():
    news = fetch_latest_news()

    if news is None:
        messagebox.showerror("Error", "Unable to fetch live news")
        return

    textbox.delete("1.0", "end")
    textbox.insert("1.0", news)

    prob = hybrid_predict(news)

    if prob >= 0.5:
        result_label.config(text="✔ REAL NEWS (LIVE)", fg="#00ff00")
    else:
        result_label.config(text="✘ FAKE NEWS (LIVE)", fg="#ff3333")


btn = tk.Button(
    root, text="Analyze News", font=("Arial", 16, "bold"), bg="#00eaff",
    fg="black", command=run_prediction
)
btn.pack()

btn_live = tk.Button(
    root, text="Fetch Live News", font=("Arial", 16, "bold"),
    bg="#ffaa00", fg="black", command=fetch_and_predict
)
btn_live.pack(pady=10)

root.mainloop()
