# Today I Learned (TIL)

Caution: This timeline is tailored for **@mohsin-ashraf** and might not be suitable for everyone.

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



