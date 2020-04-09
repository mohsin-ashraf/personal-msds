# Today I Learned (TIL)

Caution: This timeline is tailored for **@mohsin-ashraf** and might not be suitable for everyone.

## Day 09 | April 09 2020 | Tuesday
Today I started learning the common mistakes that people make while improving the machine leraning model.
- Machine Learning Diagnostic.
- Lower dimensional hypothesis functions can be plotted to check them.
- Splitting data in Training, Validation and test set.
- R<sup>2<sup>, Plotting regression error function.
- Percision, Recall, F1-score, AUC-ROC and Accuracy.
	- Precision Recall Trade off.
		- Increasing or decreasing the confidence of the estimator for postive class.
- Bias & Variance analysis.
	- Plotting training and validation error.
	- Bias occurs when the training and validation errors are colse to each other but also are very high.
	- Variance occurs when the training and validation error has a large difference and the training error is very low where as the validation error is very high.
- Regularization overcomes overfitting problem by penalizing the coefficients of the features.
	- Larger value of regularization can introduce higher bias.
	- Smaller value of regularization can introduce higher variance.
- Learning curves.
	- Training error vs Cross validation error.
- Fixes for models.
	- Get more training examples (fixes high variance).
	- Try smaller set of fetures (fixes high variance, but the model might loss some important information).
	- Using lower polynomials (fixes the high variance).
	- Using higher regularization term (fixed the high variance).
	- **Using more data don't actually fixes the high bias problem**.
	- Add more features (fixes the high bias problem).
	- Using higher polynomials (fixes the high bias problem).
	- Using smaller regularization term (fixes high bias problem).
- Machine Learning System desgins.
	- Take some manual look at what errors you model is making.
	- Try to visualize the overall results of your machine learning model against the input.
	- Numerical interpretation of your model, results and data.


## Day 08 | April 08 2020 | Wednesday
Today I continues with neural networks from backpropogation to onword.
- Summing the error over all the output units for the total error of the network.
- Gradient Checking.
- Finally Wrapped my head arround back-propogation

## Day 07 | April 07 2020 | Tuesday
Today I continued with the neural netowrks 
- Some Important links.
	- [ANNs](https://brilliant.org/wiki/artificial-neural-network/)
	- [Feedforward Networks](https://brilliant.org/wiki/feedforward-neural-networks/#formal-definition)
	- [Backpropogation](https://brilliant.org/wiki/backpropagation/)
	- [Neural Networks from scratch](http://neuralnetworksanddeeplearning.com) 
	- [Partial Derivatives](https://brilliant.org/wiki/partial-derivatives/)
- Multiclass classification neural networks output [N,1] dimensional vector where N is number of classes.
- Cost function for neural networks (sigmoid)
- Back propogation Handwritten Execution

## Day 06 | Aprit 06 2020 | Monday
Today I started learning neural networks from scratch.
- History of Neural Networks and anology with brain
- Seeing through your [tongue](https://www.youtube.com/watch?v=48evjcN73rw)!
- A single neuron is a linear classifier
- Concepts of Layers (Input, hidden, output)
- Forward propagation
- Neural Networks for multi-class classification


## Weekend Hacks&Techs | April 04-2020 / April 05-2020 | Saturday & Sunday
For weekend Hats&Techs I started Pyspark for big data
-PySpark is the Python API written in python to support Apache Spark. Apache Spark is a distributed framework that can handle Big Data analysis
- Big Data is generally greater than or equal to 100GB in size
- Single/Local Machine vs Distributed Systems

|Local/Single Machine |Distributed Systems|
|-----|-----|
|A single machine like your personal computer|A distributed system is a system whose components are located on different networked computers, which communicate and coordinate their actions by passing messages to one another|
|Local machine will use computation resources of a single machine|A distributed system has access to the computational resouces across a number of machines connected through a network|
|Very difficult or some times impossible to scale | Easily scalable by adding as many machines to the network as you want|
|Not fault tolerant, if your machine goes down your process is lost|They also include fault tolerance, if one machine fails, the whole network can still go on|
|Time consumption is very high| You can reduce the time consumption by adding more machines to the network|
- Pyspark setup [instructions](https://github.com/mohsin-ashraf/personal-msds-1/wiki/Spark-setup-on-AWS-EC2)
- Amazon EMR (Elastic Map Reduce)
	- One master node
	- Rest are core nodes
	- Access/Security group settings with IPs
- Spark DataFrames
	- [Important links for Pyspark](https://spark.apache.org/docs/)
		- Select the version and read the documentation
	- To work with PySpark DataFrames we have to use SparkSession
	- PySpark DataFrames
		- DataFrame SQL queries filteration
		- DataFrame basics
		- DataFrame basic operations
		- DataFrame Groupby and Aggregates
		- DataFrame Missing Data
		- Dates and Timestamp
	- Final DataFrame Project
		- Revised all the previous concepts and applied new techniques.
- Spark MLlib
	- Spark MLlib needs the data in a different format. You need to use **Vectors** and **VectorAssembler**
	- PySpark Pipeline
	- PySpark Linear Regression
	- PySpark Logistic Regression
	- PySpark Decision Trees
	- PySpark Random Forests
	- PySpark KMeans Clustering
	- Recommender Systems
		- Content Based
			- A Content-based recommendation system tries to recommend items to users based on their profile. The user's profile revolves around that user's preferences and tastes
		- Colaborative Filtering
			- Recommends based on the knowledge of users' attitude to items, that is, it uses the **wisdom of the crowd** to recommend items 
		- PySpark ALS Recommendation algorithm
	- Natural Language Processing using PySpark
		- CountVectorizer
		- IDF
		- Hash_TF
		- Tokenizer & RegexTokenizer
		- StringIndexer
	- Pyspark Streaming


## Day 05 | April 05 2020 | Sunday
Today I continued with weekend **#Hacks&Techs**. I also learned about the model evaluation methods in detail.
- [Regression Evaluation](https://stats.stackexchange.com/questions/89239/evaluating-a-regression-model/221311#221311?s=a757f4a4fb7a40c7a7f7d53527ac628a)
- [Classification Evaluation](https://web.archive.org/web/20150826060649/http://webdocs.cs.ualberta.ca:80/~eisner/measures.html)
- [AUC - ROC curve](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5) is used in binary classification.
	- [Another link](https://www.dataschool.io/roc-curves-and-auc-explained/)

## Day 04 | April 04 2020 | Saturday
Today I started coding Gradient Descent and implimented a very simple Gradient Descent algorithm. I did exercises using Linear Regression and uploaded the code on codebase. I also worked on Weekend **#Hacks&Techs** and did the boilerplate setup.
- Implimentation of Simplest Gradient Descent Algorithm.
- Predicting house prices using Linear Regression Single Feature.
- Predicting house prices using Linear Regression Single Feature Polynomial Regression.
- Predicting house prices using Linear Regression Multiple Features.
- Predicting house prices using Linear Regression Multiple Features Polynomial Regression.
- Predicting house price using Ridge Regression Multiple Features Polynomial Regression.
- Predicting house price using Lasso Regression Multiple Features Polynomial Regression.


## Day 03 | April 03 2020 | Friday
Today I started learning binary/multi class classification using Logistic Regression. I also learned about the Regularization and its advantages. 
- Sigmoid Function.
- Probabilistic interpretation of sigmoid function.
- Multiple Feature Logistic Regression weights.
	- Finding weights via graphical representation of the data (for atmost 2D data).
	- Finding weights via algebric method.
- Decision boundary
	- Decision boundary is the property of hypothesis function.
	- Multiple Feature, Polynomial Logistic Regression can even make complex circular decision boundaries.
- Logrithmic Cost function for Logistic Regression.
- Applying Gradient Decent on Logistic Regression.
- Advanced Optimization Algorithms.
	- Conjugate Gradient
	- BFGS (Broyden–Fletcher–Goldfarb–Shanno)
	- L-BFGS (L-Broyden–Fletcher–Goldfarb–Shanno)
	- Advantages of these Advanced Optimization Algorithms.
		- No need to manually tune **α**.
		- Often faster than Gradient Descent.
	- Dis-advantages of these Advanced Optimization Algorithms.
		- More complex than Grandient Descent.
- Multiclass Logistic Regression.
	- Combination of binary class Logistic Regressor classifiers for multiclass classification.
- Overfitting:
	- Overfitting occurs when a machine learning model fits the training data very well , but fails to generalize to new examples.
- Regularization is used to stop overfitting.
	- Mathematical intuition of Regularization.
	- Types of Regularization.
		- l1 Regularization (Lasso)
			- It penalizes the coefficients of the features such that, those coefficients exactly become zero. This type of regularization help us in feature selection. 
		- l2 Regularization (Ridge)
			- It reduces the coefficients of the features but don't make them exactly zero.
	- Regularization in Gradient Descent.
	- Regularization in Normal Equation.


## Day 02 | April 02 2020 | Thursday
Today I started digging deeper into the Gradient Descent and started learning maths behind it. I learned about Linear regression with single variable and multiple variable and how Gradient Descent minimizes the cost funciton with high dimensions. Also I refreshed my Linear Algebra concepts for the course.
- Negative & Positive slopes.
- At any minima the slope is 0.
- Linear Algebra.
	- Convex function only have one minima which is global.
	- Matrix scaler operations, Addition and multiplication.
	- Matrix Multiplicaton and its properties.
	- Matrix Inverse and Transpose.
- Linear Regression for Multiple Features.
	- Gradient Descent for multiple features Linear Regression.
	- Gradient Descent converge quickly when features are scaled.
	- Effect of learning rate on convergence.
	- Plotting Cost function to check if the Gradient Descent is working right or not.
	- Polynomial Features (Feature Interactions).
- Linear Regression for Multiple Features Normal Equation.
	- Normal Equation for Linear Regression .
	- Feature Scaling is not required.

|Gradient Decent|Normal Equation|
|-----|-----|
|Need to choose Learning Rate **α**|No need to choose Learning Rate **α**|
|Needs many iterations|Don't need to iterate|
|Works well even when **n** (dataset size) is large|Need to perform intensive matrix operations|


	
## Day 01 | April 01 2020 | Wednesday 
- Started Machine Learning Course by Adrew Ng [Coursera].
- Learned what is machine learning and types of Machine Learning.
	- Supervised Learning.
	- Unsupervised Learning.
	- Re-enforcement Learning.
	- Clustering.
	- Cocktail party problem (algorithm).
- Learned the difference between Supervised and Unsupervised Learning.
	- Supervised Learning.
		- Spam Filtering, Perdicting house prices
	- Unsupervised Learning.
		- Customer Segmentation, Article clustering
- Linear Regression.
- Machine Learning Glossary.
	- Features
	- Labels
	- Hypothesis function	
	- Cost functions 
	- Cost function
	- Contour plots for cost funtions.
	- Gradient Descent.
	- Learning Rate.



