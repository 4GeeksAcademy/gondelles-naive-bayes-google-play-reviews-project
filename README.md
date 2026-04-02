# Google Play Reviews Sentiment Classification with Naive Bayes

## Project overview

This project builds a sentiment classifier for Google Play app reviews using a simple NLP pipeline and different Naive Bayes implementations.

The goal is to predict whether a review is negative (`0`) or positive (`1`) based on its text.

The workflow follows a straightforward bootcamp-style structure:

- load the dataset
- review the available variables
- clean the text
- split into train and test sets
- vectorize reviews with `CountVectorizer`
- train and compare `GaussianNB`, `MultinomialNB`, and `BernoulliNB`
- tune the best candidate
- save the final model and vectorizer
- compare against an alternative model

---

## Dataset

The dataset contains Google Play reviews with three columns:

- `package_name`: app identifier
- `review`: review text
- `polarity`: target variable (`0` = negative, `1` = positive)

For the baseline model, the most relevant feature is the review text itself, so the main modeling pipeline uses:

- `review_clean` as input
- `polarity` as target

---

## Project structure

```text
.
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
│   ├── bernoulli_nb_tuned.pkl
│   └── vectorizer.pkl
├── src/
│   ├── app.py
│   ├── explore.ipynb
│   └── utils.py
├── README.md
└── requirements.txt