<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Objectives" data-toc-modified-id="Objectives-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Objectives</a></span></li><li><span><a href="#Model-Tuning" data-toc-modified-id="Model-Tuning-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Model Tuning</a></span><ul class="toc-item"><li><span><a href="#Hyperparameters" data-toc-modified-id="Hyperparameters-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Hyperparameters</a></span><ul class="toc-item"><li><span><a href="#Difference-from-Parametric-/-Non-Parametric-Models" data-toc-modified-id="Difference-from-Parametric-/-Non-Parametric-Models-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>Difference from Parametric / Non-Parametric Models</a></span></li></ul></li><li><span><a href="#Data-Example" data-toc-modified-id="Data-Example-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Data Example</a></span><ul class="toc-item"><li><span><a href="#Data-Prep" data-toc-modified-id="Data-Prep-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Data Prep</a></span><ul class="toc-item"><li><span><a href="#Preparing-the-Test-Set" data-toc-modified-id="Preparing-the-Test-Set-2.2.1.1"><span class="toc-item-num">2.2.1.1&nbsp;&nbsp;</span>Preparing the Test Set</a></span></li></ul></li><li><span><a href="#Trying-Different-Models-&amp;-Values" data-toc-modified-id="Trying-Different-Models-&amp;-Values-2.2.2"><span class="toc-item-num">2.2.2&nbsp;&nbsp;</span>Trying Different Models &amp; Values</a></span><ul class="toc-item"><li><span><a href="#Decision-Tree" data-toc-modified-id="Decision-Tree-2.2.2.1"><span class="toc-item-num">2.2.2.1&nbsp;&nbsp;</span>Decision Tree</a></span></li><li><span><a href="#Random-Forest" data-toc-modified-id="Random-Forest-2.2.2.2"><span class="toc-item-num">2.2.2.2&nbsp;&nbsp;</span>Random Forest</a></span></li></ul></li></ul></li></ul></li><li><span><a href="#Automatically-Searching-with-Grid-Search" data-toc-modified-id="Automatically-Searching-with-Grid-Search-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Automatically Searching with Grid Search</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#GridSearchCV" data-toc-modified-id="GridSearchCV-3.0.1"><span class="toc-item-num">3.0.1&nbsp;&nbsp;</span><code>GridSearchCV</code></a></span></li><li><span><a href="#Choice-of-Grid-Values" data-toc-modified-id="Choice-of-Grid-Values-3.0.2"><span class="toc-item-num">3.0.2&nbsp;&nbsp;</span>Choice of Grid Values</a></span></li><li><span><a href="#Exercise" data-toc-modified-id="Exercise-3.0.3"><span class="toc-item-num">3.0.3&nbsp;&nbsp;</span>Exercise</a></span></li></ul></li></ul></li><li><span><a href="#Better-Process:-Pipelines" data-toc-modified-id="Better-Process:-Pipelines-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Better Process: Pipelines</a></span><ul class="toc-item"><li><span><a href="#Advantages-of-Pipeline" data-toc-modified-id="Advantages-of-Pipeline-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Advantages of <code>Pipeline</code></a></span><ul class="toc-item"><li><span><a href="#Reduces-Complexity" data-toc-modified-id="Reduces-Complexity-4.1.1"><span class="toc-item-num">4.1.1&nbsp;&nbsp;</span>Reduces Complexity</a></span></li><li><span><a href="#Convenient" data-toc-modified-id="Convenient-4.1.2"><span class="toc-item-num">4.1.2&nbsp;&nbsp;</span>Convenient</a></span></li><li><span><a href="#Flexible" data-toc-modified-id="Flexible-4.1.3"><span class="toc-item-num">4.1.3&nbsp;&nbsp;</span>Flexible</a></span></li><li><span><a href="#Prevent-Mistakes" data-toc-modified-id="Prevent-Mistakes-4.1.4"><span class="toc-item-num">4.1.4&nbsp;&nbsp;</span>Prevent Mistakes</a></span></li></ul></li><li><span><a href="#Example-of-Using-Pipeline" data-toc-modified-id="Example-of-Using-Pipeline-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Example of Using <code>Pipeline</code></a></span><ul class="toc-item"><li><span><a href="#Without-the-Pipeline-class" data-toc-modified-id="Without-the-Pipeline-class-4.2.1"><span class="toc-item-num">4.2.1&nbsp;&nbsp;</span>Without the Pipeline class</a></span></li><li><span><a href="#With-Pipeline-Class" data-toc-modified-id="With-Pipeline-Class-4.2.2"><span class="toc-item-num">4.2.2&nbsp;&nbsp;</span>With <code>Pipeline</code> Class</a></span></li></ul></li><li><span><a href="#Grid-Searching-a-Pipeline" data-toc-modified-id="Grid-Searching-a-Pipeline-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Grid Searching a Pipeline</a></span><ul class="toc-item"><li><span><a href="#Using-ColumnTransformer" data-toc-modified-id="Using-ColumnTransformer-4.3.1"><span class="toc-item-num">4.3.1&nbsp;&nbsp;</span>Using <code>ColumnTransformer</code></a></span></li></ul></li><li><span><a href="#A-Note-on-Data-Leakage" data-toc-modified-id="A-Note-on-Data-Leakage-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>A Note on Data Leakage</a></span><ul class="toc-item"><li><span><a href="#Example-of-leaking-information" data-toc-modified-id="Example-of-leaking-information-4.4.1"><span class="toc-item-num">4.4.1&nbsp;&nbsp;</span>Example of leaking information</a></span></li><li><span><a href="#Example-of-Grid-Search-with-no-leakage" data-toc-modified-id="Example-of-Grid-Search-with-no-leakage-4.4.2"><span class="toc-item-num">4.4.2&nbsp;&nbsp;</span>Example of Grid Search with no leakage</a></span></li></ul></li></ul></li><li><span><a href="#Grid-Search-Exercise" data-toc-modified-id="Grid-Search-Exercise-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Grid Search Exercise</a></span></li><li><span><a href="#Level-Up:-Random-Searching" data-toc-modified-id="Level-Up:-Random-Searching-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Level Up: Random Searching</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#RandomizedSearchCV-with-LogisticRegression" data-toc-modified-id="RandomizedSearchCV-with-LogisticRegression-6.0.1"><span class="toc-item-num">6.0.1&nbsp;&nbsp;</span><code>RandomizedSearchCV</code> with <code>LogisticRegression</code></a></span></li></ul></li></ul></li><li><span><a href="#Level-Up:-SMOTE" data-toc-modified-id="Level-Up:-SMOTE-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Level Up: SMOTE</a></span></li></ul></div>

# Objectives

- Explain what hyperparameters are
- Describe the purpose of grid searching
- Implement grid searching for the purposes of model optimization
- Understand the Data Science pipeline



# Model Tuning

![](https://imgs.xkcd.com/comics/machine_learning.png)

## Hyperparameters

Many of the models we have looked at are really *families* of models in the sense that they make use of **hyperparameters**. Hyperparameters are also called *tuning parameters,* *meta parameters*, or *free parameters*.

While the term hyperparameter is used in a particular way within data science, **parameter** is used ambiguously. Parameter has a broad meaning that includes hyperparameters. Parameter is also used in a narrow sense that can be stated in contrast to a hyperparameter.

In general, a parameter is a term in a mathematical function that is not an argument of the function, but it can still take a range of values. For example, the equation of the regression line

$$y = \beta_0 + \beta_1  x$$ 

has $x$ and $y$ as the variables, but the *y*-intercept $\beta_0$ and the slope $\beta_1$ are the parameters of the model.

A hyperparameter is a parameter that is used to control the learning process of the algorithm. The hyperparameter values cannot be estimated from the data. (The regression model above does not have any hyperparameters.) The purpose of this lecture is to understand how to pick hyperparameters, i.e. tune the model, and implement this process into the data science pipeline.



> We can think of these **hyperparameters** as _radio dials_ of the base model. Since our model is a representation of reality, it will always have static, but tuning the hyperparametrs will allow us to have the clearist signal.



Sometimes the term parameter is used in opposition to the term hyperparameter. In this specific usuage, a model parameter is learned from the training data set.

Here are some examples of parameters and their corresponding models that we are already familiar with:

*   The estimated probabilities for a variable are parameters in logistic regression
*   The Gini impurity is a parameter for decision trees

In this strict usage of the word parameter, we say that parameters result from training and hyperparameters from tuning.

Typically it should be clear from context whether *parameter* is being used in the broad or narrow sense.

We are going to look at model turning, i.e. tuning hyperparameters, with the decision tree algorithm that we saw earlier as well as an extension of that model called *random forests*. Recall that a decision tree allows us to make

- a classifier that branches according to information gain
- a classifier that branches according to Gini impurity
- a regressor that branches according to mean squared error

A random forests, as the name suggests, is an esemble of decision trees. We'll see the details of them momentarily.

It is natural to experiment via model tuning with different values of these hyperparameters to try to improve model performance. This experimentation is part of the reason that data science is a science.

### Difference from Parametric / Nonparametric Models

Just as there is a difference between parameters (in the narrow sense) and hyperparameters, there is related a distinction between parametric and nonparametric models.

A model is parametric if the mathematical model is assumed before hand. Linear and logistic regression are both parametric models since the data is assumed to fit a particular mathematical function. For parametric algorithms, we start with a given model *form* and we then search for the optimal parameters to fill in that form.

For example, with linear regression we start with this form
$$y = \beta_0 + \beta_1  x$$ 
and find the optimal $\beta_i$'s, which are its parameters.

Since the relationship between the inputs and outputs is defined, parametric models typically are faster and easier to run than nonparametric models. As such, parametric models work well for when the input data is well-defined and predictable.

Nonparametric models are not based on a mathematical model, but rather learn the mathematical model from the data. These models are flexible and work well when the input data that is not well-defiend or complex, however they can be computationally expensive. Decision trees are an example of a nonparametric model.

Thus, somewhat confusingly, models that have hyperparameters are called nonparametric models (even though they have parameters!) and those that do not have hyperparameters are called parametric.

## Data Example

![Penguins](https://raw.githubusercontent.com/allisonhorst/palmerpenguins/69530276d74b99df81cc385f4e95c644da69ebfa/man/figures/lter_penguins.png)

> Images source: @allison_horst [github.com/allisonhorst/penguins](github.com/allisonhorst/penguins)

![Bill length & depth](https://raw.githubusercontent.com/allisonhorst/palmerpenguins/69530276d74b99df81cc385f4e95c644da69ebfa/man/figures/culmen_depth.png)

> Images source: @allison_horst [github.com/allisonhorst/penguins](github.com/allisonhorst/penguins)

### Data Prep

We'll try to predict species given the other columns' values. Let's dummy-out `island` and `sex`:

#### Preparing the Test Set

### Trying Different Models & Values

#### Decision Tree

##### Changing the branching criterion

#### Random Forest

A random forest is a generalization of decision trees and as such is called an **ensemble method**.

The key idea behind ensemble methods is the same as the *wisdom of crowds*.

An example of the wisdom of crowds is the follwoing. If you ask 100 people individually how many jelly beans are in a jar, you'll get a wide variety of answers; however, if you take the mean of those 100 answers, you'll get a number that is close to the actual value.

To continue the wisdom of the crowds metaphor for machine learning. If you have 100 different decision trees that are each trained on a different random subset of the training data, you get 100 predictions. The class that gets the most votes is the ensemble prediction.

From this idea, we get a random forest. Happily since we are using `scikit-learn` the syntax is consistent with what we are used to for decision trees.

To create a random forest in `scikit-learn`, we use `RandomForestCLassifer()`, which is in the `ensemble` module. The [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) for the random forest is up to the usual excellent standards of `scikit-learn`.

While there are over a dozen parameters for `RandomForestCLassifer()`, we'll focus on a few key ones.
*   "n_estimator" is the number of trees in the forest
*   the function to measure the quality of the split is the "criterion"
*   "max_depth" is how far the tree can grow down
*   setting "random_state" to a particular value allows us to reproduce our results

We cab see that the random forest has performed a little better than a single decision tree.

# Automatically Searching with Grid Search

It's not a bad idea to experiment with the values of your models' hyperparameters a bit as you're getting a feel for your models' performance. But there are more systematic ways of going about the search for optimal hyperparameters. One method of hyperparameter tuning is **grid searching**. 

The idea is to build multiple models with different hyperparameter values and then see which one performs the best. The hyperparameters and the values to try form a sort of *grid* along which we are looking for the best performance. For example, here is a snapshot of the grid that we'll be creating.


    4           | 'gini'    | 'auto'       | 100
    4           | 'gini'    | 'sqrt'       | 100
    4           | 'gini'    | 'log2'       | 100
    4           | 'entropy' | 'auto'       | 100
    4           | 'entropy' | 'sqrt'       | 100
    4           | 'entropy' | 'log2'       | 100
    4           | 'gini'    | 'auto'       | 200
    4           | 'gini'    | 'sqrt'       | 200
    4           | 'gini'    | 'log2'       | 200
    4           | 'entropy' | 'auto'       | 200
    4           | 'entropy' | 'sqrt'       | 200
    4           | 'entropy' | 'log2'       | 200
    ...         |   ...     |   ...        | ...
    8           | 'entropy' | 'log2'       | 200
    ___________________________________________________________
    max_depth   | criterion | max_features | n_estimators

Below, we'll see how to see the essential elements in the grid as well as where we can see all of the different combination that the alogirithm is trying.

Scikit-Learn has a [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) class whose `fit()` method runs this procedure. Note that this can be quite computationally expensive since:

- A model is constructed for each combination of hyperparameter values that we input; and
- Each model is cross-validated.

### `GridSearchCV`

**Question: How many models will we be constructing with this grid?**

*Hint: you'll be able to confirm your answer below after running the cross-validation.*

How does the above score compare to the decision tree? To the random forest we did without cross-validation?

### Choice of Grid Values

Which values should you pick for your grid? Intuitively, you should try both "large" and "small" values, but of course what counts as large and small will really depend on the type of hyperparameter.

- For a decision tree model, what counts as a small `max_depth` will really depend on the size of your training data. A `max_depth` of 5 would likely have little effect on a very small dataset but, at the same time, it would probably significantly decrease the variance of a model where the dataset is large.
- For a logistic regression's regularization constant, you may want to try a set of values that are exponentially separated, like \[1, 10, 100, 1000\].
- **If a grid search finds optimal values at the ends of your hyperparameter ranges, you might try another grid search with more extreme values.**

### Exercise

Do a grid search on a **decision tree model** of penguin species. What are the optimal values for the hyperparameters you've chosen?


```python

ct = DecisionTreeClassifier()

tree_param_grid = { 
    'min_samples_leaf': [1, 3, 5],
    'max_depth' : [4, 5, 6, 7, 8],
    'criterion' :['gini', 'entropy']
}

gs = GridSearchCV(estimator=ct, param_grid=tree_param_grid)
```


```python

gs.fit(X_train_clean, y_train)
```


```python

gs.score(X_test_clean, y_test)
```

If your decision tree performs as well as your random forest, then by Occam's Razor it is best to pick the simplest model, i.e. the decision tree.

# Better Process: Pipelines

> **Pipelines** can keep our code neat and clean all the way from gathering & cleaning our data, to creating models & fine-tuning them!

![](https://imgs.xkcd.com/comics/data_pipeline.png)

The `Pipeline` class from [Scikit-Learn's API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) is especially convenient since it allows us to use our other Estimators that we know and love!

## Advantages of `Pipeline`

### Reduces Complexity

> You can focus on particular parts of the pipeline one at a time and debug or adjust parts as needed.

### Convenient

> The pipeline summarizes your fine-detail steps. That way you can focus on the big-picture aspects.

### Flexible

> You can use pipelines with different models and with GridSearch.

### Prevent Mistakes

> We can focus on one section at a time.
>
> We also can ensure data leakage between our training and doesn't occur between our training dataset and validation/testing datasets!

## Example of Using `Pipeline`

### Without the Pipeline class

> Note that if we were to add more steps in this process, we'd have to change both the *training* and *testing* processes.

### With `Pipeline` Class

> If we need to change our process, we change it _just once_ in the Pipeline

## Grid Searching a Pipeline: Back to Penguins

> Let's first get our data prepared like we did before

> Intermediary step to treat categorical and numerical data differently

### Using `ColumnTransformer`

> Finally showing we can fit the full pipeline

> Performing grid search on the full pipeline.
Copy and paste the result from `CV_rfc.param_grid`.

## A Note on Data Leakage

Note we still have to be careful in performing a grid search!

We can accidentally "leak" information by doing transformations with the **whole data set**, instead of just the **training set**!

### Example of leaking information

### Example of Grid Search with no leakage

# Grid Search Exercise

Use a classifier (or should we try a regressor here?) of your choice to predict the category of price range for the phones in this dataset. Try tuning some hyperparameters using a grid search, and then write up a short paragraph about your findings.


```python

phones_train.head()
```


```python

# The test set has no price_range column, but it does have an ID column that we'll need to drop.

phones_test.head()
```


```python

phones_train['price_range'].value_counts()
```


```python

# We might try a regressor here since we suspect that the price ranges are ordered.
from sklearn.tree import DecisionTreeRegressor
```


```python

rt = DecisionTreeRegressor()

X = phones_train.drop('price_range', axis=1)
y = phones_train['price_range']
```


```python

rt.fit(X, y)

rt.score(X, y)
```


```python

param_grid_phone = {'max_depth': [None, 5, 10],
                   'criterion': ['mse', 'friedman_mse', 'mae']}
```


```python

gs_phone = GridSearchCV(estimator=rt, param_grid=param_grid_phone)

gs_phone.fit(X, y)
```


```python

gs_phone.predict(phones_test.drop('id', axis=1)).round()
```


```python
 
# For model metrics the best we can do, without ground-truth labels, is to compare different models' predictions.

print(gs_phone.predict(phones_test.drop('id', axis=1)).round()[20:30])
print(rt.predict(phones_test.drop('id', axis=1)).round()[20:30])
```

# Level Up: Random Searching

It is also possible to search for good hyperparameter values randomly. This is a nice choice if computation time is an issue or if you are tuning over continuous hyperparameters.

### `RandomizedSearchCV` with `LogisticRegression`

# Level Up: SMOTE

Often we encounter a problem of imbalance classification that there are too few observations of the minority class for a model to effectively learn the decision boundary. Unbalannced data sets affect a number of machine learning techniques as Logistic Regression, Decision Trees, and Random Forests.

One way to solve this problem is to **oversample** the observations in the minority class (or alternatively **undersample** the observations in the majority class) by synthesizing new observation from the minority class.

The most widely used approach to synthesizing new observations is called the **Synthetic Minority Oversampling Technique**, or [**SMOTE**](https://arxiv.org/abs/1106.1813) for short. 


Since we are leveling up, we'll *really* level-up, but introducing another machine learning technique, viz. [*k*-nearest neighbors (KNN) algorithm](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761).

Before getting into the example, please note the following,

1. Oversampling process is based on **k-nearest neighbors** of the minority class.
2. Oversampling only works with **numerical predictors** since the synthetic observations are created based on the k-nearest neighbors algorithm, which is a distance based algorithm.  

The KNN algorithm is an supervised learning algorithm like Linear Regression, Decision Trees, Random Forests, and Neural Networks. Further, the algorithm can be used to solve both regression and classification problems.

The essential idea behind this algorithm is that it assumes that similar things exist in close proximity. In the image below, we can see that similar data points are generally close to one another.

![](https://commons.wikimedia.org/wiki/File:Map1NNReducedDataSet.png)

![](https://upload.wikimedia.org/wikipedia/commons/e/e9/Map1NNReducedDataSet.png)

The [KNN algorithm](https://medium.com/swlh/k-nearest-neighbor-ca2593d7a3c4) is important in its own right, and can be implemented using our old friend [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).

We use the Scikit-Learn's breast cancer dataset to demonstrate the use of SMOTE from imblearn package.

Note that we have an imbalanced class for the target variable in this dataset.

After the oversampling and undersampling process, we observe a more balance class distribution in the target variable.
