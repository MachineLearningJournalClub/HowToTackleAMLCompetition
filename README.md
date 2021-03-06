# How to Tackle a ML Competition


## Competition Mechanics (Syllabus)

Reference : [https://www.coursera.org/learn/competitive-data-science/home/welcome](https://www.coursera.org/learn/competitive-data-science/home/welcome) 

Starting Approx. on 1st week of August 2020

- [Prerequisites](https://www.notion.so/How-to-Tackle-a-ML-Competition-7d71c740945f48f88bc50e7866179a72#bbafa83d50c54994a9cb121ee60a6704)
- [Lecture 1 : Feature Processing](https://www.notion.so/How-to-Tackle-a-ML-Competition-7d71c740945f48f88bc50e7866179a72#835d36df04e247d79b4809b2013346c6)
- [Lecture 2 : Exploratory Data Analysis (EDA)](https://www.notion.so/How-to-Tackle-a-ML-Competition-7d71c740945f48f88bc50e7866179a72#ab71dc5a74314fc6a608e5f89ac22a93)
- [Lecture 3 : Validation and Overfitting revisited](https://www.notion.so/How-to-Tackle-a-ML-Competition-7d71c740945f48f88bc50e7866179a72#addad0b9985343b78986cacad0f7a380)
- [~~Lecture 4 : Data Leakages](https://www.notion.so/How-to-Tackle-a-ML-Competition-7d71c740945f48f88bc50e7866179a72#646e2d4f5b614f50bdee5a97ba80f634)  -~~Cancellata Completamente
- [Lecture 4 : Metrics Optimization](https://www.notion.so/How-to-Tackle-a-ML-Competition-7d71c740945f48f88bc50e7866179a72#a9a18e34a87c4bb89a00a489a07f5075)
- [Lecture 5 :  Mean Encodings](https://www.notion.so/How-to-Tackle-a-ML-Competition-7d71c740945f48f88bc50e7866179a72#b8074c48fb714d5e95782d98918c674b)
- [Lecture 6: Hyperparameters Tuning](https://www.notion.so/How-to-Tackle-a-ML-Competition-7d71c740945f48f88bc50e7866179a72#66e20dacec9a4e188db33b5f4a761cb9)
- [Lecture 7: Advanced Features](https://www.notion.so/How-to-Tackle-a-ML-Competition-7d71c740945f48f88bc50e7866179a72#f189e914785d46aab5cbaca04d8418dd)
- [Lecture 8: Ensembling](https://www.notion.so/How-to-Tackle-a-ML-Competition-7d71c740945f48f88bc50e7866179a72#4bdc3f64cda946818c3225a251ce5ab3)

---

- [Code Samples](https://www.notion.so/How-to-Tackle-a-ML-Competition-7d71c740945f48f88bc50e7866179a72#e4a3348adbc94adab73fefd15856d948) (QUI CI SONO I NOTEBOOK)
- [Some more links](https://www.notion.so/How-to-Tackle-a-ML-Competition-7d71c740945f48f88bc50e7866179a72#b42c26e345ab46b8a1fbc70bd5decbb4)
- [Some more libs](https://www.notion.so/How-to-Tackle-a-ML-Competition-7d71c740945f48f88bc50e7866179a72#7886d6b40b8a4e2884800477e7027dec)

---

---

---

# Prerequisites

(Se non vi dovesse aprire qualcuno di questi articoli, accedete con la modalità "in incognito" del vostro browser e dovrebbe andare)

- Random Forest (& Decision Trees) : [https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76](https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76)
- K Nearest Neighbors : [https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)
- XGBoost : [https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7](https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7)

# Lecture 1 : Feature Processing [Simo]

Videos : [https://drive.google.com/folderview?id=1__WFpqBAQWJ-YS80cRKv_qOYQ_8y1E1u](https://drive.google.com/folderview?id=1__WFpqBAQWJ-YS80cRKv_qOYQ_8y1E1u) (con la mail di Unito avete accesso immediato) 

( notebook [https://github.com/MLJCUnito/ProjectX2020/blob/master/HowToTackleAMLCompetition/Lecture1|FeatureProcessing.ipynb](https://github.com/MLJCUnito/ProjectX2020/blob/master/HowToTackleAMLCompetition/Lecture1%7CFeatureProcessing.ipynb))

Intro : feature, physics, and scaling —>  In physics we usually have some hints on how to "preprocess" features, i.e. we have some law...

### Overview of ML Approaches (Recap)

1. Linear Models
2. Tree-Based Models
3. k Nearest Neighbors (kNN)
4. Neural Networks
5. No Free Lunch Theorem

---

### Feature Preprocessing

Lecture: Numeric Features, Categorical and Ordinal Features, Datetime and Coordinates, Handling Missing Values

Additional Material and Links: 

- [Preprocessing in Scikit-Learn](https://scikit-learn.org/stable/modules/preprocessing.html)
- [About Feature Scaling and Normalization](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html) - Sebastian Raschka
- [Gradient Descent and Feature Scaling](https://www.coursera.org/learn/machine-learning/lecture/xx3Da/gradient-descent-in-practice-i-feature-scaling) - Andrew NG
- [How to Engineer Features, Feature Engineering](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)
- [Best Practices in Feature Engineering](https://www.quora.com/What-are-some-best-practices-in-Feature-Engineering)

---

### Feature Extraction from Images / Text

Text:

- Text Preprocessing: lowercase, lemmatization, stemming, stopwords
- Bag of Words (sklearn CountVectorizer), PostProcessing: TfIdf (Term frequencies Inverse document frequencies) , N-Grams
- Embeddings (Word2Vec, Glove, FastText, Doc2Vec) : vectorizing text King-Queen Man-Woman —> Pretrained Models

Images:

- CNNs for feature extraction, Images —> Vectors; Descriptors (output from different layers)
- Finetuning & Training from Scratch
- Augmentation (Cropping, Rotating etc..)

Additional Material & Links:

**Bag of words (Text)**

- [Feature extraction from text with Sklearn](http://scikit-learn.org/stable/modules/feature_extraction.html)
- [More examples of using Sklearn](https://andhint.github.io/machine-learning/nlp/Feature-Extraction-From-Text/)

**Word2vec (Text)** 

- [Tutorial to Word2vec](https://www.tensorflow.org/tutorials/word2vec)
- [Tutorial to word2vec usage](https://rare-technologies.com/word2vec-tutorial/)
- [Text Classification With Word2Vec](http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/)
- [Introduction to Word Embedding Models with Word2Vec](https://taylorwhitten.github.io/blog/word2vec)

NLP Libraries

- [NLTK](http://www.nltk.org/)
- [TextBlob](https://github.com/sloria/TextBlob)

**Pretrained models (Images)**

- [Using pretrained models in Keras](https://keras.io/applications/)
- [Image classification with a pre-trained deep neural network](https://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11)

**Finetuning (Images)**

- [How to Retrain Inception's Final Layer for New Categories in Tensorflow](https://www.tensorflow.org/tutorials/image_retraining)
- [Fine-tuning Deep Learning Models in Keras](https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html)

---

# Lecture 2 : Exploratory Data Analysis [Jacopoison]

## Exploratory Data Analysis (EDA)

It allows you to better understand the data: Do EDA first, don't immediately dig into modeling. 

Main tools:

- Visualizations: Patterns lead to questions
- Exploiting Leaks

A few steps in EDA:

- Get domain knowledge
- Check if data is intuitive, possible misinterpretations
- Understand how the data was generated (class imbalance, crucial for proper validation scheme)

### Exploring Anonymized Data

Sensitive data are usually hashed to avoid privacy related issues (setting up a notebook) 

1. We can try to decode (de-anonymize) the data, in a legal way (of course 🙂)  by guessing the true meaning of the features or at least guess the true type (categorical etc..) —> individual features
2.  Or else, we can try to figure out feature relations 

1. Helpful Function: df.dtypes(), df.info(), x.value_counts(), x.isnull()
- Histograms (plt.hist(x)), they aggregate data tho —> can be confusing
- Index vs Values Plot
- Feature statistics with pandas —> df.describe()
- Features Mean vs Feature (by sorting this plot you can find relations —> in a data augmentation framework it might be useful)

2. Plots in order to explore feature relations → Visualizations are our art tools in the art of EDA

- plt.scatter(x1, x2)
- pd.scatter_matrix(df)
- df.corr(), plt.matshow()

### Dataset Cleaning (and other things to check)

The organizers could give us a fraction of objects they have or a fraction of features. And that is why we can have some issues with the data

- E.g. a feature that takes the same value in both training and test set (maybe it's a fraction of the whole amount of data that organizers have) —> since it is constant it's not useful for our purposes —> remove it
- Duplicated features , sometimes two columns are completely identical —> remove one (traintest.T.drop_duplicates())
- Duplicated categorical features, features are identical but rows have different names (let's rename levels of features and find duplicated)

```python
for f in categorical_features:
	traintest[f] = traintest[f].factorize()

traintest.T.drop_duplicates()
```

! We need to do label encoding correctly !

- Same with rows —> but one more question: why do have duplicated rows? A bug in the generation process?!
- Check if dataset is shuffled ! (very useful) if not there might be  data leakage. We can plot a feature vs row index and additionally smooth values with rolling average techniques
- One more time —> Visualize every possible thing —> Cool viz lead to cool features  😉

Additional Material & Links:  

**Visualization tools**

- [Seaborn](https://seaborn.pydata.org/)
- [Plotly](https://plot.ly/python/)
- [Bokeh](https://github.com/bokeh/bokeh)
- [ggplot](http://ggplot.yhathq.com/)
- [Graph visualization with NetworkX](https://networkx.github.io/)

**Others (Advanced)**

- [Biclustering algorithms for sorting corrplots](http://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_biclustering.html)

---

### Springleaf competition EDA

Notebookssss

---

### NumerAi Competition EDA

Notebooksss too

- Hardcore EDA —> sorted correlation matrix

---

*You've probably noticed that it's much about Reverse Engineering & Creativity* 

---

# Lecture 3: Validation & Overfitting Revisited[Ari]

(Notebook:   )

(Videos: [https://drive.google.com/drive/folders/18jgjbsvMGBLWpuCmFCR6f2CaZK-900QQ?usp=sharing](https://drive.google.com/drive/folders/18jgjbsvMGBLWpuCmFCR6f2CaZK-900QQ?usp=sharing))

### Validation and Overfitting

- Train, Validation & Test (Public + Private) Sets
- Underfitting vs Overfitting Recap
- Overfitting in general ≠ Overfitting in Competitions

---

### Validation Strategies

- Holdout: ngroups = 1 (sklearn.model_selection.ShuffleSplit)
- K-Fold: ngroups = k (sklearn.model_selection.KFold) and difference between K-Fold and K times Holdout
- Leave-one-out: ngroups = len(train) (sklearn.model_selection.LeaveOneOut), useful if we have too little data
- Stratification : a random split can sometimes fail, we need a way to ensure similar target distribution over different folds

The main rule you should know — *never use data you train on to measure the quality of your model*. The trick is to split all your data into *training* and *validation* parts.

Below you will find several ways to validate a model.

- ****Holdout scheme:**
    1. Split train data into two parts:  partA and  partB
    2. Fit the model on partA, predict for partB
    3. Use predictions for partB  for estimating model quality. Find such hyper-parameters that quality on partB  is maximized.
- ****K-Fold scheme:**
    1. Split train data into K folds.
    2. Use the predictions to calculate quality on each fold. Find such hyper-parameters, that quality on each fold is maximized. You can also estimate mean and variance of the loss. This is very helpful in order to understand significance of improvement.
- ****LOO (Leave-One-Out) scheme:**
    1. Iterate over samples: retrain the model on all samples except current sample, predict for the current sample. You will need to retrain the model N times (if N is the number of samples in the dataset).
    2. In the end you will get LOO predictions for every sample in the trainset and can calculate loss.

Notice, that these validation schemes are supposed to be used to estimate the quality of the model. When you've found the right hyper-parameters and want to get test predictions don't forget to retrain your model using all training data.

---

### Data Splitting Strategies

Setup validation to mimic train / test split. E.g. time series, we need to rely on the time trend instead of randomly picking up values —> Time Based Splits 

Different splitting strategies can differ significantly

1. In generated features
2. In a way the model will rely on that features
3. In some kind of target leak 

Splitting Data into Train and Validation 

- **Random, Rowwise** ; Most common, we assume that rows are independent from each other
- **Timewise**, we generally have everything before some date in the train-set and everything after in the test-set (e.g. Moving window validation)
- **By ID**, id can be a unique identifier of something
- **Combined**, combining some of the above mentioned

Logic of feature generation depends on the data splitting strategy. 

---

### Problems Occurring During Validation

Validation problems (usually caused by inconsistency of  data):

- Validation Stage (e.g. if we are predicting sales we should take a look at holidays, so there's a reason to expect some particular behavior)
- Submission Stage (e.g. LeaderBoard (LB) score is consistently higher/lower that validation score, LB score is not correlated with validation score at all) —> Leaderboard Probing (calculate mean for train data and try to probe the test data distribution by submitting .. 🤯)

    What if we have imbalanced distributions? We should ensure the same distribution in test and validation (again, by LB probing) 

LB Shuffle: it happens when positions on Public and Private LB are drastically different, main reasons:

- Randomness (main main reason)
- Little amount of data
- Different Public & Private distributions

Additional Material & Links: 

- [Validation in Sklearn](http://scikit-learn.org/stable/modules/cross_validation.html)
- [Advices on validation in a competition](http://www.chioka.in/how-to-select-your-final-models-in-a-kaggle-competitio/)

---

# ~~Lecture 4: Data Leakages (or how to cheat)~~

(Not sure to do it, it really depends on the competition, if it is Kaggle-like we should take a look at leaks)

### Basic Data Leaks

1. Leaks in time series: Split should be done on time 
    - In real life we don't have information from the future
    - Check train, public and private splits. If one of them is not on time you've found a data leak
    - Even when split by time, features may contain information about future: User history in CTR tasks; Weather
2. Unexpected information:
    - Metadata
    - Information on IDs
    - Row Order

---

### Leaderboard Probing

- Categories tightly connected with "id" are vulnerable to LB probing
- Adapting global mean via LB probing
- Some competition with data leakages: Truly Native; Expedia; Flavours of Physics
- Pairwise tasks, data leakage in item frequencies

---

### Case Study: Expedia Kaggle Competition

---

Additional Material & Links: 

- [Perfect score tips on Kaggle](https://www.kaggle.com/olegtrott/the-perfect-score-script)
- [Leakage in Competitions](https://www.kaggle.com/docs/competitions#leakage)
- [Something more on data leakage](https://www.kaggle.com/dansbecker/data-leakage)

---

# Lecture 4 : Metrics Optimization [Pio]

### Metrics:

- Why there are so many
- Why should we care about them in competitions
- Loss vs Metric
- Review the most important metrics
- Optimization techniques for the metrics

Metrics are an essential part of any competition, they are used to evaluate our submissions. Why do we have different metrics for each competition? There are different ways to measure the quality of an algorithm 

- E.g. How to formalize effectiveness for an online shop ? It can be the number of times the website was visited or the number of times something was ordered using this website

Chosen metric determines optimal decision boundary 

![How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-11_alle_16.39.16.png](How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-11_alle_16.39.16.png)

If your model is scored with some metric, you get best results by optimizing exactly that metric 

With LB probing we can check if the train and test sets have some incongruences with respect distributions, we gotta be careful wrt metrics optimization if there's some imbalance

---

### Regression Metrics

(Add Notebook)

- MSE, RMSE, R-Squared
- MAE
- MSPE, MAPE
- MSLE

MSE: Mean Square Error 

![How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-11_alle_16.49.32.png](How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-11_alle_16.49.32.png)

RMSE: Root Mean Square Error

![How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-11_alle_16.52.37.png](How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-11_alle_16.52.37.png)

R-Squared: 

![How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-11_alle_16.56.24.png](How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-11_alle_16.56.24.png)

MAE: Mean Absolute Error

![How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-11_alle_16.57.55.png](How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-11_alle_16.57.55.png)

MSPE : Mean Square Percentage Error

![How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-12_alle_19.21.32.png](How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-12_alle_19.21.32.png)

MAPE: Mean Absolute Percentage Error

![How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-12_alle_19.21.41.png](How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-12_alle_19.21.41.png)

(R)MSLE: Root Mean Square Logarithmic Error

![How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-12_alle_19.25.07.png](How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-12_alle_19.25.07.png)

---

### Classification Metrics

Accuracy : How frequently our class prediction is correct

![How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-12_alle_19.31.21.png](How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-12_alle_19.31.21.png)

- Best constant : predict the most frequent class —> dummy example : Dataset of 10 cats and 90 dogs —> Always predicts  dog and we get 90% accuracy, the baseline accuracy could be very high even if the result is not correct

Log Loss : Logarithmic Loss 

![How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-12_alle_19.36.20.png](How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-12_alle_19.36.20.png)

AUC ROC : Area Under Curve 

Cohen's Kappa (& Weighted Kappa)

![How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-12_alle_19.44.56.png](How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-12_alle_19.44.56.png)

![How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-12_alle_19.47.22.png](How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-12_alle_19.47.22.png)

---

### Approaches for Metrics Optimization

Loss and Metric

- Target Metric : is what we want to optimize (e.g Accuracy)
- Optimization loss : is what our model optimizes

![How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/0329e18bd1cb4cf1167fc37d20e5488f.jpg](How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/0329e18bd1cb4cf1167fc37d20e5488f.jpg)

Approaches for target metric optimization

- Just run the right model (lol!) : MSE, Logloss
- Preprocess train and optimize another metric : MSPE, MAPE, RMSLE
- Optimize another metric, postprocess predictions: Accuracy, Kappa
- Write Custom Loss Function (if you can)
- Optimize another metric, use early stopping

---

Probability Calibration

- Platt Scaling : Just fit Logistic Regression to your prediction
- Isotonic Regression
- Stacking  : Just fit XGBoost or neural net to your predictions

---

Additional Material & Links  : 

[](http://people.cs.bris.ac.uk/~flach/papers/Performance-AAAI19.pdf)

[DAIgnosis: Exploring the Space of Metrics](https://medium.com/mljcunito/daignosis-exploring-the-space-of-metrics-c6bca5d53acb)

Classification

- [Evaluation Metrics for Classification Problems: Quick Examples + References](http://queirozf.com/entries/evaluation-metrics-for-classification-quick-examples-references)
- [Decision Trees: “Gini” vs. “Entropy” criteria](https://www.garysieling.com/blog/sklearn-gini-vs-entropy-criteria)
- [Understanding ROC curves](http://www.navan.name/roc/)

Ranking

- [Learning to Rank using Gradient Descent](http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf) -- original paper about pairwise method for AUC optimization
- [Overview of further developments of RankNet](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)
- [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/) (implemtations for the 2 papers from above)
- [Learning to Rank Overview](https://wellecks.wordpress.com/2015/01/15/learning-to-rank-overview)

Clustering

- [Evaluation metrics for clustering](http://nlp.uned.es/docs/amigo2007a.pdf)

---

# Lecture 5 : Mean Encodings[?]

(We can follow this one —> [https://www.kaggle.com/vprokopev/mean-likelihood-encodings-a-comprehensive-study](https://www.kaggle.com/vprokopev/mean-likelihood-encodings-a-comprehensive-study)) 

The general idea of this technique is to add new variables based on some feature. In simplest cases, we encode each level of a categorical variable with the corresponding target mean

Why does it work ? 

- Label encoding gives random order.. No correlation with target
- Mean encoding helps to separate zeros from ones

It turns out that this sorting quality of mean encoding is quite helpful. Remember, what is the most popular and effective way to solve machine learning problem? Is grading using trees (XGBoost). One of the few downsides is an inability to handle high cardinality categorical variables. Trees have limited depth, with mean encoding, we can compensate it!

---

### Regularization

(Notebook) 

- Cross Validation inside training data (CV loop)

    Usually decent results with 4-5 folds 

- Smoothing
- Adding random noise
- Sorting and calculating expanding mean

---

### Extensions and generalizations

- Regression and multiclass
- Many-to-many relations
- Time Series
- Interactions and numerical features

---

# Lecture 6: Hyperparameter Tuning & Advanced Features[?]

### How do we tune hyperparameters?

- Select the most influential parameters

    There are tons of params and we can't tune all of them 

- Understand, how exactly they influence the training
- Tune them

    Manually (change and examine)

    Automatically (hyperopt, grid search etc..) —> some libraries : Hyperopt; Scikit-optimize; Spearmint; GPyOpt; RoBo; SMAC3 

    We need to define a function that specifies all the params and a search space, the range for the paramas where we want to look for the solution. 

- Different values for params can lead to 3 behaviors:
    1. Underfitting (bad)
    2. Good Fit and Generalization (good)
    3. Overfitting (bad)

Color-Coding Legend

Red Parameter : 

- Increasing it impedes fitting, it reduces overfitting

Green parameter: 

- Increasing it leads to a better fit on train set, increase it if model underfits, decrease it if it overfits

### Tree-based Models

- GBDT (XGBoost & LightGBM)

    Parameters

![How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-13_alle_12.56.21.png](How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-13_alle_12.56.21.png)

- Random Forest / Extra Trees

    (Notebook on how to find sufficient n_estimators) 

    N_estimators (the higher the better) 

    max_depth (it can be unlimited)

    max_features 

    min_samples_leaf

    Others: criterion (gini etc) , random_state, n_jobs

    ---

### Neural Nets

- Number of neurons per layer
- Number of layers
- Optimizers

    SGD + Momentum 

    - Better generalization

    Adam/Adagrad ... 

    - Adaptive methods lead to more overfitting
- Batch Size
- Learning Rate, there's a connection between batch size and learning rate (proporzionalità diretta nelle dimensioni delle due)
- Regularization
    - L2 / L1 for weights
    - Dropout /Dropconnect
    - Static Dropconnect

    ---

### Linear Models

- SVC/SVR (sklearn)
- Logistic Regression + regularizers (sklearn)
- SGDClassifier /Regressor (sklearn)
- FTRL, Follow The Regularized Leader (Vowpal Wabbit) —> For the data sets that do not fit in the memory, we can use Vowpal Wabbit. It implements learning of linear models in online fashion. It only reads data row by row directly from the hard drive and never loads the whole data set in the memory. Thus, allowing to learn on a very huge data sets.

Regularization Parameter (C, alpha, lambda ...) 

### Practical Guide

- Data Loading

    Do basic preprocessing and convert csv/txt files into hdf5/npy for much faster loading 

    Don't forget that by default data is stored in 64-bit arrays, most of the times you can safely downcast it to 32-bits 

    Large datasets can be processed in chunks 

- Performance Evaluation

    Extensive validation is not always needed

    Start with fastest models  LightGBM 

    ![How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-13_alle_14.02.23.png](How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-13_alle_14.02.23.png)

- Fast & Dirty always Better

    Don't pay too much attention to code quality

    Keep things simple: save only important things 

- Initial Pipeline

    Start with simple solution 

    Debug full pipeline (from reading data to writing submission file)

- Best Practices from Software Development

    Use good variable names

    Keep your research reproducible (fix random seeds, use Version Control Systems) 

    Reuse Code 

- Read Papers !!
- My Pipeline

    Read Forums and Examine kernel first (you'll find some discussions going on)

    Start with EDA and a baseline 

    Add features in bulks 

    Hyperparameters optimization 

    Use macros for frequent code and custom libraries as well 

    ---

    Additional Material & Links: 

    - [Tuning the hyper-parameters of an estimator (sklearn)](http://scikit-learn.org/stable/modules/grid_search.html)
    - [Optimizing hyperparameters with hyperopt](http://fastml.com/optimizing-hyperparams-with-hyperopt/)
    - [Complete Guide to Parameter Tuning in Gradient Boosting (GBM) in Python](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)
    - [Far0n's framework for Kaggle competitions "kaggletils"](https://github.com/Far0n/kaggletils)
    - [28 Jupyter Notebook tips, tricks and shortcuts](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/)

    ---

# Lecture 7: Advanced Features [Simo]

### Statistics and distance based features

- Statistics on initial features
- Neighbors (kNN, Bray - Curtis metric etc)

- Matrix Factorizations for Feature Extraction (NMF, SVD, PCA)

    Pay attention to apply the same transformation to all your data (concatenate train & test and apply PCA or whatever)

- Feature Interactions

    Sums, Diffs, Multiplications, Divisions

    ![How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-13_alle_14.23.07.png](How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-13_alle_14.23.07.png)

- t-SNE , UMAP (Manifold Learning Methods)

    Interpretation of hyperparameters (e.g. Perplexity)

---

Additional Material & Links:

Matrix Factorization:

- [Overview of Matrix Decomposition methods (sklearn)](http://scikit-learn.org/stable/modules/decomposition.html)

t-SNE:

- [Multicore t-SNE implementation](https://github.com/DmitryUlyanov/Multicore-TSNE)
- [Comparison of Manifold Learning methods (sklearn)](http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html)
- [How to Use t-SNE Effectively (distill.pub blog)](https://distill.pub/2016/misread-tsne/)
- [tSNE homepage (Laurens van der Maaten)](https://lvdmaaten.github.io/tsne/)
- [Example: tSNE with different perplexities (sklearn)](http://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html#sphx-glr-auto-examples-manifold-plot-t-sne-perplexity-py)

Interactions:

- [Facebook Research's paper about extracting categorical features from trees](https://research.fb.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/)
- [Example: Feature transformations with ensembles of trees (sklearn)](http://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html)

---

# Lecture 8: Ensembling [Ari]

Prendiamo da qui —> ([https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205#](https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205#:~:text=Stacking%20mainly%20differ%20from%20bagging%20and%20boosting%20on%20two%20points.&text=Second%2C%20stacking%20learns%20to%20combine,weak%20learners%20following%20deterministic%20algorithms))

An ensemble method combines the predictions of many individual classifiers by majority voting.

Ensemble of *low-correlating* classifiers with slightly greater than 50% accuracy will outperform each of the classifiers individually.

![https://polakowo.io/datadocs/assets/apes.jpg](https://polakowo.io/datadocs/assets/apes.jpg)

Condorcet's jury theorem:

- If each member of the jury (of size N) makes an *independent* judgement and the probability p of the correct decision by each juror is more than 0.5, then the probability of the correct decision PN by the majority m tends to one. On the other hand, if p<0.5 for each juror, then the probability tends to zero.

    ![How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-16_alle_16.45.12.png](How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-16_alle_16.45.12.png)

- where m as a minimal number of jurors that would make a majority.
- But real votes are not independent, and do not have uniform probabilities.

Uncorrelated submissions clearly do better when ensembled than correlated submissions.

Majority votes make most sense when the evaluation metric requires hard predictions.

Choose bagging for base models with high variance.

Choose boosting for base models with high bias.

Use averaging, voting or rank averaging on manually-selected well-performing ensembles.

[KAGGLE ENSEMBLING GUIDE](https://mlwave.com/kaggle-ensembling-guide/)

### **Averaging**

- Averaging is taking the mean of individual model predictions.
- Averaging predictions often reduces variance (as bagging does).
- It’s a fairly trivial technique that results in easy, sizeable performance improvements.
- Averaging exactly the same linear regressions won't give any penalty.
- An often heard shorthand for this on Kaggle is "bagging submissions".

**Weighted averaging**

- Use weighted averaging to give a better model more weight in a vote.
- A very small number of parameters rarely lead to overfitting.
- It is faster to implement and to run.
- It does not make sense to explore weights individually (α+β≠1) for:

    α+β≠1)

    - AUC: For any α, β, dividing the predictions by α+β will not change AUC.
    - Accuracy (implemented with argmax): Similarly to AUC, argmax position will not change.

**Conditional averaging**

- Use conditional averaging to cancel out erroneous ranges of individual estimators.
- Can be automatically learned by boosting trees and stacking.

---

### **Bagging**

- Bagging (bootstrap aggregating) considers *homogeneous* models, learns them independently from each other in parallel, and combines them following some kind of deterministic averaging process.
- Bagging combines *strong learners* together in order to "smooth out" their predictions and reduce variance.
- Bootstrapping allows to fit models that are roughly independent.

![https://polakowo.io/datadocs/assets/Ozone.png](https://polakowo.io/datadocs/assets/Ozone.png)

- The procedure is as follows:
    - Create N random sub-samples (with replacement) for the dataset of size N.
    - Fit a base model on each sample.
    - Average predictions from all models.
- Can be used with any type of method as a base model.
- Bagging is effective on small datasets.
- Out-of-bag estimate is the mean estimate of the base algorithms on 37% of inputs that are left out of a particular bootstrap sample.
    - Helps avoid the need for an independent validation dataset.
- Parameters to consider:
    - Random seed
    - Row sampling or bootstrapping
    - Column sampling or bootstrapping
    - Size of sample (use a much smaller sample size on a larger dataset)
    - Shuffling
    - Number of bags
    - Parallelism
- See [Tree-Based Models](https://nbviewer.jupyter.org/github/polakowo/machine-learning/blob/master/ml-notes/TreeBasedModels.ipynb)

**Bootstrapping**

- Bootstrapping is random sampling with replacement.
- With sampling with replacement, each sample unit has an equal probability of being selected.
    - Samples become approximatively independent and identically distributed (i.i.d).
    - It is a convenient way to treat a sample like a population.
- This technique allows estimation of the sampling distribution of almost any statistic using random sampling methods.
- It is a straightforward way to derive estimates of standard errors and confidence intervals for complex estimators.
- For example:
    - Select a random element from the original sample of size N and do this B times.
    - Calculate the mean of each sub-sample.
    - Obtain a 95% confidence interval around the mean estimate for the original sample.
- Two important assumptions:
    - N should be large enough to capture most of the complexity of the underlying distribution (representativity).
    - N should be large enough compared to B so that samples are not too much correlated (independence).
- An average bootstrap sample contains 63.2% of the original observations and omits 36.8%.

---

### **Boosting**

- Boosting considers *homogeneous* models, learns them sequentially in a very adaptative way (a base model depends on the previous ones) and combines them following a deterministic strategy.
- This technique is called boosting because we expect an ensemble to work much better than a single estimator.
- Sequential methods are no longer fitted independently from each others and can't be performed in parallel.
- Each new model in the ensemble focuses its efforts on the most difficult observations to fit up to now.
- Boosting combines weak learners together in order to create a strong learner with lower bias.
    - A weak learner is defined as one whose performance is at least slightly better than random chance.
    - These learners are also in general less computationally expensive to fit.

**Adaptive boosting**

- At each iteration, adaptive boosting changes the sample distribution by modifying the weights of instances.
    - It increases the weights of the wrongly predicted instances.
    - The weak learner thus focuses more on the difficult instances.
- The procedure is as follows:
    - Fit a weak learner ht with the current observations weights.

        ht

    - Estimate the learner's performance and compute its weight αt (contribution to the ensemble).

        αt

    - Update the strong learner by adding the new weak learner multiplied by its weight.
    - Compute new observations weights that expresse which observations to focus on.

        ![How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-16_alle_16.48.04.png](How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-16_alle_16.48.04.png)

- See [Tree-Based Models](https://polakowo.io/datadocs/docs/machine-learning/tree-based-models)

**Gradient boosting**

Gradient boosting doesn’t modify the sample distribution:

- At each iteration, the weak learner trains on the remaining errors (so-called pseudo-residuals) of the strong learner.

Gradient boosting doesn’t weight weak learnes according to their performance:

- The contribution of the weak learner (so-called multiplier) to the strong one is computed using gradient descent.
- The computed contribution is the one minimizing the overall error of the strong learner.

Allows optimization of an arbitrary differentiable loss function.

The procedure is as follows:

- Compute pseudo-residuals that indicate, for each observation, in which direction we would like to move.
- Fit a weak learner ht to the pseudo-residuals (negative gradient of the loss)
- Add the predictions of ht multiplied by the step size α (learning rate) to the predictions of ensemble

    ![How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-16_alle_16.48.31.png](How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Schermata_2020-07-16_alle_16.48.31.png)

- See [Tree-Based Models](https://polakowo.io/datadocs/docs/machine-learning/tree-based-models)

---

### **Stacking**

- Stacking considers *heterogeneous* models, learns them in parallel and combines them by training a meta-model to output a prediction based on the different weak models predictions.
- Stacking on a small holdout set is blending.
- Stacking with linear regression is sometimes the most effective way of stacking.
- Non-linear stacking gives surprising gains as it finds useful interactions between the original and the meta-model features.
- Feature-weighted linear stacking stacks engineered meta-features together with model predictions.
- At the end of the day you don’t know which base models will be helpful.
- Stacking allows you to use classifiers for regression problems and vice versa.
- Base models should be as diverse as possible:
    - 2-3 GBMs (one with low depth, one with medium and one with high)
    - 2-3 NNs (one deeper, one shallower)
    - 1-2 ExtraTrees/RFs (again as diverse as possible)
    - 1-2 linear models such as logistic/ridge regression
    - 1 kNN model
    - 1 factorization machine
- Use different features for different models.
- Use feature engineering:
    - Pairwise distances between meta features
    - Row-wise statistics (like mean)
    - Standard feature selection techniques
- Meta models can be shallow:
    - GBMs with small depth (2-3)
    - Linear models with high regularization
    - ExtraTrees
    - Shallow NNs (1 hidden layer)
    - kNN with BrayCurtis distance
    - A simple weighted average (find weights with bruteforce)
- Use automated stacking for complex cases to optimize:
    - CV-scores
    - Standard deviation of the CV-scores (a smaller deviation is a safer choice)
    - Complexity/memory usage and running times
    - Correlation (uncorrelated model predictions are preferred).
- Greedy forward model selection:
    - Start with a base ensemble of 3 or so good models.
    - Add a model when it increases the train set score the most.

**Multi-level stacking**

- Always do OOF predictions: you never know when you need to train a 2nd or 3rd level meta-classifier.
- Try skip connections to deeper layers.
- For 7.5 models in previous layer add 1 meta model in next layer.
- Try [StackNet](https://github.com/h2oai/pystacknet) which resembles a feedforward neural network and uses Wolpert's stacked generalization (built iteratively one layer at a time) in multiple levels to improve accuracy in machine learning problems.
- Try [Heamy](https://github.com/rushter/heamy) - a set of useful tools for competitive data science (including ensembling).

    ---

    # Code & Samples

    [https://drive.google.com/drive/folders/1CkGuP3wn9AAhEXIVEcHFKOVK9OCn3B2I?usp=sharing](https://drive.google.com/drive/folders/1CkGuP3wn9AAhEXIVEcHFKOVK9OCn3B2I?usp=sharing) (è anche qua sul drive del MLJC)

    [Notebooks.zip](How%20to%20Tackle%20a%20ML%20Competition%203c68879573b34ce58a7f701932769c9a/Notebooks.zip)

    ---

    # Some more links

    [http://ndres.me/kaggle-past-solutions/](http://ndres.me/kaggle-past-solutions/)

    [https://www.kaggle.com/python10pm/plotting-with-python-learn-80-plots-step-by-step](https://www.kaggle.com/python10pm/plotting-with-python-learn-80-plots-step-by-step) 

    [https://machinelearningmastery.com/how-to-get-started-with-deep-learning-for-time-series-forecasting-7-day-mini-course/](https://machinelearningmastery.com/how-to-get-started-with-deep-learning-for-time-series-forecasting-7-day-mini-course/)

    ---

    # Some more libs

    - [https://www.streamlit.io/](https://www.streamlit.io/)
    -