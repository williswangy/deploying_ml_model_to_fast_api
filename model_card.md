# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The prediction task is to determine whether a person makes over 50K a year. 
We use a RandomForestClassifier with optimized hyperparameters in scikit-learn 1.2.0. 
Hyperparameter tuning was performed using GridSearchCV. 
The optimal parameters used are:

 - bootstrap: True
- max_depth: 20
- min_samples_leaf: 1
- min_samples_split: 5
- n_estimators: 100


- **bootstrap**: This parameter determines whether bootstrap samples are used when building trees. If set to `True`, each tree in the ensemble is built on a bootstrap sample from the training set. If `False`, the whole dataset is used to build each tree. Bootstrap samples are a random sample of the data where sampling is done with replacement. This means that some samples may be used multiple times in one bootstrap sample, while others may not be used at all.

- **max_depth**: This parameter regulates the maximum depth of each tree in the forest. The depth of a tree is the maximum distance between the root and any leaf. A tree of depth 20 will have at most 20 layers of nodes from top to bottom. Limiting the maximum depth of the tree can help reduce overfitting.

- **min_samples_leaf**: This parameter specifies the minimum number of samples required to form a leaf node. In other words, a split in the tree will only be considered if it leaves at least `min_samples_leaf` training samples in each of the left and right branches.

- **min_samples_split**: This parameter determines the minimum number of samples required to split an internal node. For example, if `min_samples_split` is set to 5, there must be at least 5 samples at a node for a split to be attempted at that node.

- **n_estimators**: This parameter sets the number of trees in the forest. The higher the number of trees, the more complex the model can become because it will have more opportunities to learn from the data. However, a large number of trees can also increase the computational complexity and might lead to overfitting. In this case, the model is using 100 trees.

The model is saved as a pickle file in the model folder. All training steps and metrics are logged in the file "journal.log".

## Intended Use
This model has the capability to estimate the income bracket of an individual based on several characteristics. It is designed to be utilized by students, scholars, or for research purposes.

## Training Data
The Census Income Dataset, sourced from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income), is provided as a csv file. The dataset originally comprises 32,561 rows and 15 columns, which includes the target label "salary", 8 categorical attributes, and 6 numerical attributes. More specifics about each attribute can be found at the UCI link provided.

The target label "salary" falls into two categories ('<=50K', '>50K'), displaying a class imbalance with a roughly 75% to 25% split.

This dataset was divided into a training set and a test set using a 70-30 split, with stratification based on the target label "salary". In preparation for model training, a One Hot Encoder was applied to the categorical features, and a label binarizer was used on the target label.

## Evaluation Data

For the purpose of evaluating the model, 30% of the dataset was kept separate. The categorical features and the target label were each transformed using the One Hot Encoder and Label Binarizer, respectively, which were fitted based on the training set.

## Metrics

The performance of the classification model is evaluated using precision, recall, and fbeta metrics, with the confusion matrix also calculated to provide further insights.

Using the test set, the model achieves the following scores:
The model achieves the following scores using the test set:

- Precision: 0.775
- Recall: 0.653
- Fbeta: 0.709

The confusion matrix is as follows:

|                   | Predicted Negative | Predicted Positive |
|-------------------|--------------------|--------------------|
| Actual Negative   |       6971         |        446          |
| Actual Positive   |        817         |        1535        |

## Ethical Considerations

It is important to note that the dataset should not be considered a fair representation of the salary distribution across different population categories, and caution should be exercised when making assumptions about salary levels based on this dataset alone.

## Caveats and Recommendations
The dataset was extracted from the 1994 Census database, making it an outdated sample that may not accurately reflect the current population distribution. As a result, it is not suitable for use as a statistical representation of the population. However, it can still be effectively utilized for training machine learning models in classification or related tasks. It is important to note that any insights or predictions derived from this dataset should be interpreted with caution and validated with more recent and representative data sources for real-world applications.