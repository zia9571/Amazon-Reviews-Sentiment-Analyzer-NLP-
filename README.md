<img width="527" height="393" alt="download" src="https://github.com/user-attachments/assets/20ddcbb3-5fa6-4a99-ac58-1b6bffad7ee7" /># Amazon Reviews Sentiment Analyzer (NLP)

This project uses Natural Language Processing (NLP) techniques to analyze Amazon product reviews and classify them as either positive or negative. It serves as a practical demonstration of a complete data science workflow, from data cleaning and preprocessing to model training and evaluation.

## Table of Contents
- [Project Overview](#project-overview)
- [Methodology](#methodology)
- [Data Source](#data-source)
- [Dependencies](#dependencies)
- [Files in this Repository](#files-in-this-repository)
- [Results](#results)
- [How to Run the Code](#how-to-run-the-code)
- [Next Steps](#next-steps)

---

## Project Overview

The primary goal of this project is to build a machine learning model that can automatically determine the sentiment of a product review based on its text. By training a model on a large dataset of Amazon reviews, we can create a system capable of quickly gauging customer opinion, which can be valuable for market research and product development.

---

## Methodology

The project follows a standard machine learning pipeline:

1.  **Data Collection and Labeling**: The dataset was sourced from Kaggle and contains product reviews with a `Score` (rating) and `Text` (review content). Reviews with a score of 4 or 5 were labeled as **positive**, and reviews with a score of 1 or 2 were labeled as **negative**. Neutral reviews (score 3) were excluded.

2.  **Text Preprocessing**: The raw text data was cleaned and normalized to prepare it for the model. This involved:
    * Converting text to lowercase.
    * Removing punctuation and special characters.
    * Removing common English **stopwords** (e.g., 'the', 'is', 'a').
    * Applying **lemmatization** to reduce words to their base form (e.g., 'running' to 'run').

3.  **Text Vectorization**: To convert the text into a numerical format, I used the **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorizer. This technique assigns a weight to each word based on its importance in the document and across the entire dataset.

4.  **Model Training**: A **Logistic Regression** model was trained on the preprocessed and vectorized data. This model was chosen for its effectiveness and interpretability as a baseline classifier for text classification problems.

5.  **Model Evaluation**: The model's performance was evaluated on a held-out test set using key metrics such as accuracy, precision, recall, and the F1-score.

---

## Data Source

The dataset used in this project is the **Amazon Fine Food Reviews** dataset, available on Kaggle.

- **Dataset Link**: [https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

---

## Dependencies

The following libraries are required to run the notebook:
- `pandas`
- `numpy`
- `scikit-learn`
- `nltk`
- `matplotlib`
- `seaborn`
- `wordcloud`

You can install these dependencies using pip:
`pip install -r requirements.txt`

---

## Files in this Repository

- `Amazon_Reviews_Sentiment_Analyzer.ipynb`: The main Jupyter Notebook containing all the code for data cleaning, modeling, and evaluation.
- `requirements.txt`: A file listing all the necessary Python libraries.
- `README.md`: This file, providing an overview of the project.

---

## Results

The trained Logistic Regression model achieved the following performance on the test set:

- **Accuracy**: 89.2%

### Confusion Matrix
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/1f77386b-861c-43dc-8f9e-812e81b0e240" />

**Interpretation**:
- **Accuracy**: The model correctly classified 89.2% of the reviews.
- **Precision (Positive)**: When the model predicts a review is positive, it is correct 88% of the time.
- **Recall (Positive)**: The model correctly identified 92% of all actual positive reviews.
- **Confusion Matrix**: The matrix shows that the model had a higher number of false negatives (misclassifying positive reviews as negative) compared to false positives.

---

## How to Run the Code

1.  **Clone the repository**: `git clone https://github.com/your-username/Amazon-Reviews-Sentiment-Analyzer.git`
2.  **Download the dataset**: Get the `Reviews.csv` file from the Kaggle link provided above and place it in the same directory as the notebook.
3.  **Install dependencies**: Run `pip install -r requirements.txt`.
4.  **Open the notebook**: Launch Jupyter Notebook or JupyterLab and open `Amazon_Reviews_Sentiment_Analyzer.ipynb` to explore the code.

