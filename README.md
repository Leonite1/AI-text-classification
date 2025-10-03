# AI-text-classification
This project is about building a system that can classify text into different categories using AI.  
It includes everything needed to prepare data, train models, test them, and make predictions on new text.  
The goal is to make it easy to experiment with different text classification techniques.

## What the Project Does
- Takes raw text data and prepares it for machine learning.  
- Trains models to learn patterns from the text.  
- Evaluates how well the models perform.  
- Predicts the category of new, unseen text.

## Features
- ✅ Data preprocessing and cleaning pipeline  
- 🧪 Multiple model training options (ML & DL)  
- 📊 Model evaluation with standard metrics (Accuracy, Precision, Recall, F1)  
- 📝 Inference scripts for predicting labels on new text  
- 🧠 Organized folder structure for models, outputs, and source code  
- ⚡ Easy environment setup using `requirements.txt`

## Dataset
The 20 Newsgroups dataset is subject to the terms of scikit-learn.
https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset

## Structure
Open terminal at the repo root and activate venv and install deps:
pip install -r requirements.txt

Then we run Train file as a module with:
python -m src.train

To generate basic stats we run the Eda file:
python -m src.eda

We also can evaluate a saved model with:
python -m src.evaluate --model models/clf.pkl

One-off prediction using the saved model:
python -m src.predict "your text here"

Start FastAPI server:
uvicorn src.app:app --reload --port 8000
