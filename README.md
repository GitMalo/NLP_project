# Project:

## Description:
This project has for purpose to predict the rating of a review for a restaurant.

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

|   Rating   | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
|     6      |    0.62   |  0.84  |   0.72   |   347   |
|     10     |    0.11   |  0.01  |   0.01   |   149   |
|     8      |    0.00   |  0.00  |   0.00   |   5     |
|     2      |    0.40   |  0.10  |   0.16   |   233   |
|     4      |    0.00   |  0.00  |   0.00   |   7     |
|     9      |    0.44   |  0.36  |   0.40   |   491   |
|     7      |    0.00   |  0.00  |   0.00   |   14    |
|     5      |    0.62   |  0.88  |   0.73   |   745   |
|------------|-----------|--------|----------|---------|
|  Accuracy  |    0.58   |        |          |  1991   |
| Macro Avg  |    0.28   |  0.27  |   0.25   |  1991   |
| Weighted Avg |   0.51   |  0.58  |   0.52   |  1991   |

- baseline improved: in progress
I am trying to improve the baseline model. The first thing that I want to do is to give more columns of the dataset that I have identified in the data analysis part as important but for the moment I didn't find a way to do it.

- deep learning: 


