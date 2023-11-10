# Project:

## Description:
This project has for purpose to predict the rating of a review for a restaurant.
https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews

The project is divided in 5 parts:
- Data analysis (exploration of the data to see how it is structured)
- preprocessing (cleaning of the data)
- baseline (creation of a baseline model)
- baseline improved (improvement of the baseline model)
- deep learning (creation of a deep learning model)

## Progression:
- Data analysis: done
I explored the dataset to see how it is structured and what kind of data it contains.
I printed some graphs to see the distribution of the data and the features that have the most impact on the rating.
I analyzed the reviews to see how to do the preprocessing.

- preprocessing: done
I did a preprocessing program which can be reused in other projects. It works with an intruction file which contains the instructions to clean the data (json file) and the columns on which the instructions have to be applied. At the end, the program returns a dataframe with the cleaned data.
There is also a check instruction file which stops the program and explains what is wrong with the instruction file if there is a problem (wrong column name, wrong instruction, wrong argument, ...).

- baseline: done
I created a model class with a random forest classifier and a tdidf vectorizer. The results are not very good but it is a good start.

|    | Precision | Recall | F1-Score | Support |
|----|-----------|--------|----------|---------|
| 2.0 | 0.69      | 0.82   | 0.75     | 351     |
| 3.0 | 0.11      | 0.01   | 0.01     | 134     |
| 1.0 | 0.44      | 0.09   | 0.14     | 231     |
| 5.0 | 0.45      | 0.40   | 0.42     | 481     |
| 4.0 | 0.62      | 0.87   | 0.73     | 762     |
| **Accuracy** |          |        | 0.59     | 1959    |
| **Macro Avg** | 0.46      | 0.44   | 0.41     | 1959    |
| **Weighted Avg** | 0.54   | 0.59   | 0.54     | 1959    |

I developed the same model with other columns of the dataset and the results are a little better but the main changement is that the extreme classes have a better score because of the utilisation of the number of reviews and followers of the reviewer. These features have a huge impact on the extreme classes but not on the other ones.

|    | Precision | Recall | F1-Score | Support |
|----|-----------|--------|----------|---------|
| 1.0 | 0.68      | 0.81   | 0.74     | 351     |
| 2.0 | 0.00      | 0.00   | 0.00     | 134     |
| 3.0 | 0.61      | 0.08   | 0.15     | 231     |
| 4.0 | 0.49      | 0.49   | 0.49     | 481     |
| 5.0 | 0.65      | 0.86   | 0.74     | 762     |
| **Accuracy** |          |        | 0.61     | 1959    |
| **Macro Avg** | 0.49      | 0.45   | 0.42     | 1959    |
| **Weighted Avg** | 0.57   | 0.61   | 0.56     | 1959    |


- baseline improved: done
I improved the baseline, to do this I tested several baselines to see which combination of model and vectorizer is the best. After this I tried different hyperparameters and the results are the following:

Best Model: LogisticRegression
Best Parameters: {'C': 1, 'max_iter': 200, 'penalty': 'l2'}
Best Score: 0.6180576806820787

The score is not very high because I used cross validation to avoid overfitting.

- deep learning: done
I tested different deep learning models to see which one of them is the most adapted to the problem. At the end the best model was the LSTM with an accuracy of 0.8472. This model is the best because he is capable of long-term memory storage which is important to analyze reviews.


