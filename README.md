Supply Chain Analytics: Demand Forecasting of Products

Demand forecasting is the practice of predicting customers' future demand for products. This project forecasts item-level demand (the Quantity sold) for an online gift shop using a three-step machine-learning pipeline: clustering → classification → prediction.

Contents
Dataset
Data Preparation
Feature Engineering
Correlation Analysis
Modelling Approach
Models Compared
Conclusion
Known Issues and Next Steps
Getting Started
Dataset
Source: UCI Online Retail dataset
Multivariate, sequential, time-series data covering all transactions between 2010 and 2011 for a UK-based, registered, non-store online retailer (mostly gift items).
541,909 rows and 8 attributes:
Attribute	Description
InvoiceNo	Transaction identifier
StockCode	Product code
Description	Product name (free text)
Quantity	Units sold per line item (target)
InvoiceDate	Date and time of the transaction
UnitPrice	Price per unit
CustomerID	Customer identifier
Country	Customer country
Data Preparation
Description has 1,454 missing values and CustomerID has 135,080. Neither can be sensibly imputed, so rows with nulls were dropped.
Text in Description was cleaned (lower-cased, non-letters removed), tokenized, stop-word filtered, and lemmatized with NLTK.
A subset of the first 50,000 cleaned rows was used for modelling.
Categorical columns (InvoiceNo, StockCode, CustomerID, Country, Product Type, Colour_type) were label-encoded.
The data was split 80/20 into train and test sets.
Feature Engineering

New features created from the text and date columns:

Product_type: nouns extracted from Description using a part-of-speech tagger.
Colour_type: the first colour word found in Description (e.g. red, white, cream), or no_color.
Revenue: UnitPrice × Quantity.
Year, Month, Day, DayOfWeek: parsed from InvoiceDate.
Cluster number: produced by the clustering step (see below).
Correlation Analysis

A correlation heatmap was computed for the numeric attributes CustomerID, Quantity, and UnitPrice.

Show Image

(Save the heatmap image as correlation_heatmap.png in the repo root so it renders here.)

Reading the heatmap (red = +1, white = 0, blue = −1):

The diagonal is 1.0, as expected, since every variable is perfectly correlated with itself.
All off-diagonal cells are essentially white, i.e. correlations close to 0:
Quantity vs UnitPrice: no meaningful linear relationship. Price alone does not explain how many units are sold per line.
CustomerID vs Quantity / UnitPrice: near zero. This is expected, because CustomerID is an identifier, not a quantity.
Takeaway: the numeric attributes are not linearly redundant (no multicollinearity concern), but they also offer little linear signal for predicting Quantity. This motivates the non-linear, tree-based models and the engineered features (product type, colour, date parts, cluster number) used later.
Modelling Approach
Step 1: Clustering (K-Prototypes)
Challenge: the data mixes numerical and categorical attributes, and the categorical ones are important and cannot be dropped. Converting them to numbers would distort their meaning.
Solution: the K-Prototypes algorithm, which is based on the k-means paradigm but measures object similarity using both numeric and categorical attributes while keeping k-means' efficiency.
The number of clusters was chosen by plotting clustering cost for k = 2..14 and picking an elbow (k = 3).
Output: a Cluster number column for the training set.
Step 2: Classification (Linear SVM)
The cluster number from step 1 is treated as the target variable so that validation and test records can be matched to the training clusters.
Linear SVC performed best among the algorithms tried and was used to assign cluster numbers to the test set.
Step 3: Prediction (Random Forest)
All features, including Cluster number, are used with Quantity as the target.
Random Forest performed best for demand prediction. Hyperparameters were explored with RandomizedSearchCV (3-fold CV, 100 candidates; this took about 5 hours on Colab).
Models Compared

Classification of Quantity (test-set accuracy and micro-F1 reported in the notebook):

Random Forest (selected)
K-Nearest Neighbours (k = 3)
SVC (RBF kernel)
AdaBoost
Logistic Regression (one-vs-rest)
Gaussian Naive Bayes
Decision Tree
Gradient Boosting
Conclusion
The pre-processing phase prepares clean, model-ready inputs.
Feature engineering adds variables (product type, colour, revenue, date parts) that bring extra value to demand interpretation.
The three-step model (clustering, classification, prediction) lets the company visualize relationships between predictor variables and customize its forecasts.
Known Issues and Next Steps

Items spotted while reviewing the notebook that are worth fixing:

Revenue column overwritten: the type-casting cell sets X.iloc[:, 8] = X.iloc[:, 3], which replaces Revenue with a copy of UnitPrice. It should be X.iloc[:, 8] cast from itself.
Target leakage: Revenue = UnitPrice × Quantity is built from the target. If it is kept as a feature when predicting Quantity, the model can reconstruct the answer. Drop it from the prediction features.
Regression vs classification: Quantity has hundreds of distinct values but is modelled with classifiers and scored by accuracy. A regressor (e.g. RandomForestRegressor, which the tuning step already uses) evaluated with MAE/RMSE is more appropriate.
Time-aware split: a random split on time-series data leaks future information. Prefer a chronological train/test split.
Inconsistent categorical indices: the elbow search uses categorical=[0,1,4,5,6,7] while the final fit uses [0,1,4,5,6]. Make these consistent.
Gradient Boosting prediction bug: gbc is predicted with the Naive Bayes model lr instead of gb.
Subset of data: only the first 50,000 rows are used. Scaling to the full dataset or sampling randomly would give more representative results.
Ideas: add true time-series baselines (e.g. aggregate weekly demand per product and compare with ARIMA/Prophet), and report regression metrics for every model.
Getting Started
bash
git clone https://github.com/mishra-krishna/Supply-Chain-Analytics-Project
cd Supply-Chain-Analytics-Project
pip install pandas numpy scikit-learn nltk kmodes matplotlib
jupyter notebook Demand_Forecasting_of_items.ipynb

Download OnlineRetail.csv from the UCI link above and update the file path in the notebook (it currently points to a Google Drive location). NLTK resources used: punkt, wordnet, stopwords, averaged_perceptron_tagger.
