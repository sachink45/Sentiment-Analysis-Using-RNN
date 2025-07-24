# 🎬 Movie Review Sentiment Analysis using RNN

This project performs **sentiment analysis** on movie reviews using a **Recurrent Neural Network (RNN)** built with **TensorFlow/Keras**. The model classifies reviews as **positive** or **negative**, and a **Streamlit web app** provides an interactive interface for users to test the system in real time.

---

## 🧠 What This Project Does

- Trains an RNN on the **IMDB movie review dataset** to detect sentiment.
- Preprocesses and tokenizes text input using Keras tools.
- Uses an **Embedding layer** followed by an **RNN (SimpleRNN)** to capture sequence patterns in reviews.
- Exports a trained model (`sentiment_model.h5`) that can classify new text inputs.
- Provides a **Streamlit UI** to enter a review and get instant predictions.

---

## 📦 Technologies Used

| Tool         | Purpose                                |
|--------------|----------------------------------------|
| Python       | Programming language                   |
| TensorFlow/Keras | Building and training the RNN model |
| Streamlit    | Web interface for prediction           |
| IMDB Dataset | Movie reviews labeled by sentiment     |

---

## 🏗 Model Architecture

- **Embedding Layer**: Converts words to dense vectors.
- **SimpleRNN Layer**: Processes sequences of word embeddings.
- **Dense Output Layer**: Sigmoid activation for binary classification.

> Loss Function: `binary_crossentropy`  
> Optimizer: `adam`  
> Metric: `accuracy`

---

## 🚀 How to Run the Project

1. **Clone the Repository**

```bash
git clone https://github.com/sachink45/Sentiment-Analysis-Using-RNN.git

cd Sentiment-Analysis-Using-RNN

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

streamlit run web.py
