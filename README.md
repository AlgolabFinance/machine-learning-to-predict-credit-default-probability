# machine-learning-to-predict-credit-default-probability
Data cleaning and applying machine learning algorithms to predict credit default probability

The credit score is a numeric expression measuring people’s creditworthiness. It is vital for the banking system to utilize credit score as a method to support the decision-making about credit applications. Therefore the credit score is developed and applied for this purpose because it is time-saving and easily comprehensible. There are a certain number of studies in creating a scoring model with the use of machine learning. In this paper, we are going to build and compare the performance of several machine learning models in predicting the default of credit card clients.

The models that are used in our study include the traditional statistical model of logistic regression, the classical machine learning classifiers such as Support Vector Classifier, Random Forest and Boosting (Gradient and Ada Boosting) as parts of the popular machine learning package (scikit-learn). Moreover, we also add the new boosting decision tree based models of XGBoost, CATBoost and LightBGM, which are proven to be quite effective and popular in Kaggle's competition recently.

As we are studying about probability of default, the imbalanced of data is an usual problems. In this study, for each type of model, 2 methods of oversampling are applied which are the utilization of class weights and SMOTE. Furthermore, in order to ensure the stability of the models, cross-validation is also used to compare the results.

Dealing with imbalanced data As description of the dataset, there are only 6,636 out of 30,000 samples is classified as default. This is commonly known as unbalance data, which will bias the prediction model towards the more dominate class. For example, in extreme cases where 1 out of 2 classes has samples size less than 5% total population, making the prediction of the model very accurate on that train-test dataset but have no predictive power over other datasets. Therefore,we must use balancing strategies for selecting random samples to avoid majority class bias. To deal with this problem, we apply 2 popular approaches, the class-weight method and SMOTE (Synthetic Minority Oversampling Technique).

Class-weight method directly modify the loss function by giving more (or less) penalty to the classes with more (or less) weight. We want to give higher weight to minority class and lower weight to majority class. scikit-learn has a convenient utility function to calculate the weights based on class frequencies

SMOTE on the other hand is a oversampling technique that allow us to up-sample the default data (minority with value of 1) to have a balanced dataset. To discribe how it works:

SMOTE works by creating synthetic samples from the minor class (no-subscription) instead of creating copies.
SMOTE randomly choosing one of the k-nearest-neighbors and using it to create a similar, but randomly tweaked, new observations
With each below algorithms, we include repeatedly the class-weight method and SMOTE to create a balance dataset using different classifier from those algorithms

Models and performance measuremance
In this study, we select 8 classification algorithms out of the most popular techniques being applied that we want to try, includings:

Logistic regression
Random Forest
Support Vector Machine
Gradient Boosting
AdaBoost
XGBoost
CATBoost
LightBGM
These algorithms are non-linear classifiers as we assume the non-linear characteristic for credit scoring The 4-fold cross validation is also applied to effectively measure the skill of the our models on new data. We measure the precision and accuracy by averaging the 4 results. Moreover, we apply both class weight and SMOTE method to deal with imbalanced data issue as mentioned above. This means for each classification algorithm, there will be 2 tests, one use class-weight and SMOTE is applied for the other.

To measure performance of each classification algorithms, we shows 6 ratios:

Fit_time:
This is the time to run model (in seconds), showing the speed of calculation, depending on the volumne of data, numbers of variables and sophisication of the model. We will consider also the trade-off between speed and performance for those methods

Accuracy
It measures how many observations, both positive and negative, were correctly classified
𝐴𝑐𝑐𝑢𝑟𝑎𝑐𝑦=𝑡𝑝+𝑡𝑛𝑡𝑝+𝑓𝑝+𝑡𝑛+𝑓𝑛
 
However, it's not ideal to use this ratio when the data is imbalanced. The prediction will be high accurate by simply classifying all observations as the majority class.

Precision and recall
In contrast, Precision-Recall is a useful measure of success of prediction when the classes are very imbalanced. In information retrieval, precision is a measure of result relevancy, while recall is a measure of how many truly relevant results are returned. High precision relates to a low false positive rate, and high recall relates to a low false negative rate.
𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛=𝑡𝑝𝑡𝑝+𝑓𝑝
 
𝑅𝑒𝑐𝑎𝑙𝑙=𝑡𝑝𝑡𝑝+𝑓𝑛
 
F1 score
The F1 score combines precision and recall into one metric by calculating the harmonic mean between those two. For the F1 score, a threshold should be find to maximize the score.
𝐹1=2×𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛×𝑅𝑒𝑐𝑎𝑙𝑙𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛+𝑅𝑒𝑐𝑎𝑙𝑙=2𝑡𝑝2𝑡𝑝+𝑓𝑝+𝑓𝑛
 
Binary classification such as in this case should apply the F1 score

ROC_AUC
The ROC curve visualizes the tradeoff between true positive rate (TPR) and false positive rate (FPR), ROC AUC ratio means the area under the curve, ranging from 0 to 1. This ratio is more useful because it tells us that this metric shows how good at ranking predictions your model is. It tells you what is the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance.
