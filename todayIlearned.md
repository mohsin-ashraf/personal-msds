# Today I Learned (TIL)

Caution: This timeline is tailored for **@mohsin-ashraf** and might not be suitable for everyone.

## Day 16 | April 16 2020 | Thursday
Today I continued with matrix spaces and moved on to system of linear equations and matrix determinants.
- Orthogonality, Null spaces and dot-product.
- The four sub-spaces.
- Systems of equations.
	- **Ax = b**.
	- Gaussian Elimination.
		- Matrix Agumentation.
		- Upper diagonal matrix
		- Back-substitution
	- Echelon form.
		- povits of the matrix.
		- Zeros below the pivot.
	- Reduced row Echelon form.
- Matrix Determinants.
	- Only square matrices have a determinant.
	- Determinant is a single number of the matrix.
	- If the determinant of a matrix if **0** then it has linearly dependent columns.
	- Determinant of 3X3 matrix.
		- Agument the same matrix along the right/left verticle axis.
		- From the first matrix take the diagonal elements which might include elements from the agumented matrix.
		- From the second matrix take the off-diagonal elements which might include elements from the matrix which is being used to agument.
	- Finding value of elements from the determinant of the matrix.


## Day 15 | April 15 2020 | Wednesday
Today I kept on learning the matrices and went on understanding about Matrix Rank, Matrix Spaces.
- Rank is defined as dimentionality of information and is given as *r E N, s.t. 0 <= r <= min(m,n) where N = natural numbers, and m,n are the rows and columns of a matrix.*
- Another way of defining the Rank is as *The rank of a matrix is the largest number of columns (or rows) that can form a linearly independent set.*
	- Full row rank matrix
	- Full column rank matrix
	- Full matrix rank
	- Reduced rank
- rank(A) = rank(A<sup>T</sup>) = rank(A<sup>T</sup>A) = rank(AA<sup>T</sup>) *where rank is the function that gives the rank of a matrix*.
- Shifting a matrix by a scaler.	
	-*A* = **A + cI** where **A** is a square matrix, **c** is a constant and **I** is an identity matrix.
- Matrix Spaces.
	- A column space *C(A)* of a matrix **A** is the vector space that is spaned by all of the columns in the matrix doesn't matter if they are linearly independent or linearly dependent.
	- *C(A) = {B<sub>1</sub>a<sub>1</sub> + ... + B<sub>n</sub>a<sub>n</sub>}*
	- A row space *R(A) = C(A<sup>T</sup>)* of a matrix **A** is the vector space that is spaned by all the columns in the matrix doesn't matter if they are linearly independent or linearly dependent.
	- Null Spaces *N(A)* of a matrix **A** is the set of all the vectors **{V}** such that **Av = O** and **V != O**.
	- If a matrix has no null space it has linearly independent vectors and vice versa.


## Day 14 | April 14 2020 | Tuesday
Today I started learning about the matrices, notations, terminologies and their operations.
- Types of matrices.
- Trace of the matrix.
- Matrix multiplication.
- Diagonal matrix multplications.
- Consider square matrices **A**, **B** then **(AB)<sup>T</sup>** = **B<sup>T</sup>A<sup>T</sup>**.
- Column weighted multiplication and row weighted multiplication.
- Multiplicative Identity and Additive Identity.
- Matrix **S** is symetric iff **S = (A+A<sup>T</sup>)/2 and A is NxN matrix**
	- **S = S<sup>T</sup>**


## Day 13 | April 13 2020 | Monday
Today I started the Linear Algebra course and learned the fundamental concepts about the linear algebra. I explored some basic vector operations and learned about different spaces.
- Vocabulary of Linear Algebra and Nomenclature.
	- Row vectors, Column Vectors, heads, tails and  Dimensions of the vectors.
- Importance of Linear Algebra and Area of applications.
- Vectors are ordered list of numbers or functions.
- Geometeric interpretation of vectors.
- Vectors subtraction and addition.
	- Dimension needs to be the same.
	- Geometeric Interpretation of operations.
- Scaler multiplication.
	- Magnitude and direction of vector after scaler multiplication.
- Dot product of vectors.
	- A<sub>m<sub>x</sub>n</sub><sup>T</sup> . B<sub>pxq</sub> gives us R<sub>mxq</sub>
	- For dot product to occur **x** and **p** needs to be equal in both matrices.
	- DotProduct = ||A||.||B|| . Cos(angle_between_A_and_B)
	- Law of cosine.
	- Dot product is commutative.
- Magnitude/Norm of vector.
	- || V || = Square_root (V<sup>T</sup>V)
	- Pythagoras theorm.
- Hadamard multiplication or element wise multiplication.
- Outer product ==> V<sub>mxn</sub>.W<sub>pxq</sub><sup>T</sup> ==> R<sub>mxp</sub> (this is the difference between dot product and outer product).
- Complex Number vectors.
	- Complex Number multiplication.
- Harmitian Transpose (a.k.a. conjugate transpose).
- Unit vectors.
- Dimensions and Fields.
- Subspace and Ambient space.
	- Subspace ==> Linear combination ==> Scaler multiplication and addition.
	- Subspace must be closed under addition and/or scaler multiplication.
	- Contains the ZERO vector.
- Ambient space is a higher dimentional space in which sub-spaces lie.
- Span is all possible linear combinations of the vectors in the set.
- Linearly Independent set of vectors is a set in which no vector can be generated by the linaer combination of other vectors.
- Linearly dependent if **O** = C<sub>1</sub>V<sub>1</sub> + C<sub>2</sub>V<sub>2</sub>  + C<sub>3</sub>V<sub>3</sub> ... C<sub>n</sub>V<sub>n</sub>  where C is element of **R** and C != 0.
	- That is if the any linear combination of the vectors creates the ZERO vector.
- Basis Vectors.
	- Dimensionality reduction.

## Day 12 | April 12 2020 | Sunday
Today I started learning Anomaly detection problem and Recommendation systems and finally completed the course on Machine learning.
- Anomaly detection.
	- Normal Distribution parameterized with mean and std.
	- Each feature contributes its Normal density function in order to detect anomalies.
	- Don't use anomalies in your training data.
	- Anomaly detection evaluation.
		- Precision Recall
		- F<sub>1</sub>-Score
		- Confussion matrix
	- Anomaly detection vs Supervised Learning.
	- Feature selection in Anomaly detection.
		- Check the distribution of your data if its Normal distribution then its an important feature. (use visualization of data)
		- Try to transform features with non-normal distribution into normal distribution using some transformation function.
		- Feature Engineering.
		- Get more feature or generate new features
	- Multivariate Gaussian (normal) Distribution.
		- Using covariance matrix to change the distribution of the features.
		- Changing the mean of the feature distribution.
		- Can be computationally expensive.
- Recommendation Systems.
	- Motivation for recommendation systems.
	- Feature vectors for data.
	- Weight vector for each user for recommendation.
	- Collaborative filtering.
		- Learn features and their values for the data, from the user preferences.
		- Iterative back and forth learning of weight vectors and feature vectors.
		- Vectorized Implimentation of collaborative filtering.
		- Mean normalization.
- Large Scale Machine Learning.
	- Always do a sanity check for large datasets by using a smaller dataset.
		- See if it has a lot more variance to capture or it just has captured all the variance of the data and furhter data is not improving anymore. Then we might not need to train the data on full dataset.
	- Use stochastic gradient descent for large datasets to improve from every single exmple.
	- Shaffle the datasets for using stochastic nature optimizers
	- Mini-batch gradient descent.
	- Continous monitering of cost function if its decreasing or not.
	- Online learning as the data stream is comming in your model starts to learn.
	- Parallelsim and MapReduce.
- Machine Learning Pipelines.
	- Measuring accuracy of each step of the pipeline.
	- Accuracy of the pipeline steps can affect the model accuracy.


## Day 11 | April 11 2020 | Saturday
Today I started unsupervised machine learning and learned about the clustering and dimensionality reduction.
- k-Means clustering.
	- Cluster assignment (K = number of clusters)
	- Cluster movement
	- Objective function for K-Means clustering
	- Local Optimal points in K-Means Clustering
		- Try iterative run of K-Means algorithm using random initializations
		- Caculate the distortion of the clusters with the points
		- Pick the best point
	- Chosing the number of clusters
		- Visualize the data.
		- Elbow method using the distortion
- Dimentionality Reduction.
	- Data Compression
	- Data visualization
	- Speed-up machine learning algorithms and reduces the required computation
	- Converts highly corelated features (with almost linear relationship) to a single feature using projection.
	- Projection of higher dimensional data to lower dimensional data (e.g. R<sup>n</sup> ==> R<sup>n-1</sup> ==> R<sup>n-2</sup> ...)
- Principal Component Analysis (PCA).
	- Calculates a lower dimensional space for higher dimentional data, such that the squared distances of the data points are small with the calculated lower dimentional space (reduces the projection error).
	- Always perform feature scaling while using PCA (mostly mean normalization is used).
	- Eigen vectors of Covariance vectors of the features of the data.
	- Caculate the eigen vectors from covariance matrix and multiply it with the data keeping the K components (K = [1,2 ... n]).
	- The more the varaince you are able to capture in lower dimensions the better (best case upto 99%).
	- Reconstruction from the compressed data to original data.
- Using PCA to reduce overfitting is a bad idea although might work okay but it would be better to use regularization to overcome the overfitting, Since PCA through aways some of the variance/information of the data.
	

## Day 10 | April 10 2020 | Friday
Today I started learning the Support Vector Machines.
- Cost function for support vector machines.
	- Essentially the cost funciont adjusts itself such that the decision boundary gets wider using the margins.
	- These margins grow from either side of the boundary equally.
	- Intuitive understanding of why only support vectors contributes in the decision boundary of the support vector machine.
- Kernal functions.
	- Calculation new features using the landmarks.
	- Type of kernal functions.
		- Polynomial kernal (x<sup>T</sup>.l + constant)<sup>degree</sup>.
			- Used very rarely.
		- Linear kernal.
		- Gaussian kernal.
		- String kernal (if the input data is text).
		- chi-square kernal
		- histogram kernal.
		- Intersection kernal.
- SVM in practice.
	- When using Gaussian Kernal do perform feature scaling.
	- Multi-class classification SVMs


## Day 09 | April 09 2020 | Thursday
Today I started learning the common mistakes that people make while improving the machine leraning model.
- Machine Learning Diagnostic.
- Lower dimensional hypothesis functions can be plotted to check them.
- Splitting data in Training, Validation and test set.
- R<sup>2</sup> , Plotting regression error function.
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
For weekend Hacks&Techs I started Pyspark for big data
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



