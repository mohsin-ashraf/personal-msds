# Today I Learned (TIL)

Caution: This timeline is tailored for **@mohsin-ashraf** and might not be suitable for everyone.

## Day 44 | May 14 2020 | Thursday
Today I completed the second module of the deep learning specialization part five Sequence models and learned the following topics.
- Negative Sampling.
	- Pair of positive and negative samples.
	- Binary classifiers of vocab size (a layer of binary classifiers with vocab_size units), with **K** classifiers learning at a time.
- Glove word vectors.
	- X<sub>i,j</sub> = Number of times the word **i** (target) appears in the context of word **j** (context).
	- minimize Sum<sup>m</sup><sub>i</sub>Sum<sup>m</sup><sub>j</sub> f(X<sub>i,j</sub>)(theta<sub>i</sub><sup>T</sup>e<sub>j</sub> + b<sub>i</sub>+b<sub>j</sub><sup>'</sup>  - log(X<sub>i,j</sub>))<sup>2</sup>.
- Sentiment Classification using embeddings.
- Debiasing the word embeddings.
	- Identify bias direction.
	- Neutralize: For every word that is not difinitional, project to get rid of bias.
	- Equalize pairs.


## Day 43 | May 13 2020 | Wednesday
Today I started the second module of deep learning specialization part five Sequence models and learned the following topics.
- Featurized representation of words: Words Embeddings.
- Learning a feature vector against words, so that, these features can be used to know the semantic of a word.
	- Similary objects will have very similar representation e.g *apple ~= orange*.
- Visualization of word embeddings by reducing its dimensions (plotting it will clearly show the relationship between similar objects).
- Using pre-trained words embeddings (on let say 1B-100B words) for transfer learning.
- Continue finetuning the word embeddings with new data (if you have large dataset optional).
- Word Embeddings are quite usefull for the tasks like summarization and Named entity recognition.
- Word Embeddings are less usefull for the tasks like Language modeling and Machine translation.
- Properties of word embeddings.
	- Learn anology reasoning e.g man --> women as king --> *Queen*.
	- Cosine Similarity ==> Sim(U,V) = (U<sup>T</sup>V)/(||U|| * ||V||).
- Embedding matrix.
	- Selecting a perticular word column from embedding matrix using one hot vector of that perticular word.
- Learning word embeddings.
	- Taking word embeddings of a sentence and feeding it to the networks (to maybe predict the next word).
		- These embeddings can sometimes when stacked together can get thousands of dimensional vector, hence to reduce the dimensionality only a window of few previous words is used (let say 5-10 previous words).
		- You can also take constant number of words from before and after the target word.
- Skip gram model.
	- Randomly taking an input word and then learning a maping to some target word.
	- Iterating the above process for a few times with the same input word.


## Day 42 | May 12 2020 | Tuesday
Today I continued with the deep learning specialization part five Sequence Models completed the first module and its quizzes and assignments.
- Gated Recurrent Unit.
	- Helps in capturing the long term dependencies and overcomes the problem of vanishing gradient.
	- Memory unit.
- Long Short Term Memory (LSTM).	
- Bi-directional RNN (BRNN).
	- Has both forward and backward components.
	- y<sup>'</sup> = g (W<sub>y</sub>[a<sup> --> < t ></sup>,a<sup> <-- < t ></sup>] + b<sub>y</sub>)
- Deep RNNs.


## Day 41 | May 11 2020 | Monday
Today I started deep learning specialization part five Sequence models and started its first module.
- Motivation for sequence models.
- Notations for sequence models.
- Why not regular neural networks for sequence models.
	- Inputs and Outputs can be of different lengths in different examples.
	- Doesn't share features leaned across different positions of the text.
- At each time stamp the Recurrent neural network passes its activations to the next time stamp.
- Forward propogation is Recurrent Neural networks.
	- a<sup>< t ></sup> = g(W<sub>aa</sub>a<sup>< t - 1></sup> + W<sub>ax</sub>X<sup> < t ></sup> + b<sub>a</sub>)
	- y<sup>'< t ></sup> = g (W<sub>ya</sub>a<sup>< t ></sup> + b<sub>y</sub>)
- Backpropogation through time.
	- Logistic Loss function
- Different types of RNNs.
	- Many to many architecture (Sequence to Sequence)
	- Many to one architecture (Sequence to one number)
	- One to One architecture
	- One to many architecture
	- Encoder and Decoder
- Language modeling and sequence generation.
	- Large corpus of the language.
	- Probablistic estimations of the words appearing given previous words.
	- P(y<sup>< T ></sup>) = P (y<sup>< T ></sup> | y<sup>< 1 ></sup>,y<sup>< 2 ></sup>,...,y<sup>< T - 1 ></sup>)
- Sampling novel sequences.
- Vanishing/Exploding Gradients with RNNs.
	- Long term dependencies in languages.
	- Gradient clipping for gradient exploding.


## Day 40 | May 10 2020 | Sunday
Today I completed the certification of deep learning specialization part four by completing all the required assignments and quizzes. The relevant certificate can be found [here](https://www.coursera.org/account/accomplishments/certificate/ZZUFQ8NFYQTY)


## Day 39 | May 9 2020 | Saturday
Today I started working on the programming exercises and quizzes of the deep learning specialization part 4 and completed exercises and quizes of first two modules.


## Day 38 | May 8 2020 | Friday
Today I completed the certification of deep learning specialization part three by completing all the required assignments and quizzes. The relevant certificate can be found [here](https://www.coursera.org/account/accomplishments/certificate/KBJ9H37EG6RL)


## Day 37 | May 7 2020 | Thursday
Today I completed the certification of deep learning specialization part two by completing all the required assignments and quizzes. The relevant certificate can be found [here](https://www.coursera.org/account/accomplishments/certificate/JKHPHS87S7M4)


## Day 36 | May 6 2020 | Wednesday
Today I completed the certification of deep learning specialization part one by completing all the required assignments and quizzes. The relevant certificate can be found [here](https://www.coursera.org/account/accomplishments/certificate/WR79VC6AWN63)


## Day 35 | May 5 2020 | Tuesday
Today I started working on the programming exercises and quizes of the deep learning specialization part 1 and completed exercises and quizes of first two modules.


## Day 34 | May 4 2020 | Monday
Today I revisited the neural style transfer from deep learning specialization part 4 to clear up a couple of things which were left unclear last time.


## Day 33 | May 3 2020 | Sunday
Finally I completed the deep learning specialization part 4 for convoutional neural networks.
- One-shot learning problem.
	- Recognize the person given one single image.
	- Neural network learns the similarity function.
		- f(image_1,image_2) = degree of difference between images.
		- f(image_1,image_2) > threshold then the persons are same otherwise different.
- Siamese network.
	- Outputs the encoding of the input images.
	- If the images are of the same person the difference between the encodings must be small.
	- Apply the backpropogation to learn the function.
- Triplet loss function.
	- Looking at 3 images (examples).
		- One image is sample, the other is positive example (image of the same person) and third one is a negative example (image of another person.)
	- || f(A) - f(P) ||<sup>2</sup> + alpha <= ||f(A) - f(N) ||<sup>2</sup>
	- Loss function.
		- L(A,P,N) = max(||f(A)-f(P)||<sup>2</sup> - ||f(A) - f(N)||<sup>2</sup> + alpha , 0)
	- Data set need pairing.
		- Pair(A,P) similar and Pair(A,N) different.
- Face verification.
	- Use siamese networks for face verification with sigmoid unit in the last layer.
	- Chi-squared similarity.
- Neural style transfer.
	- Generate a new image (G) from content image (C) and style image (S).
	- Features learnt by the convolutional layers of the networks.
	- Cost function for NST (neural style transfer).
		- J(G) = alpha * J(C,G) + beta * J(S,G).
		- Apply gradient descent on this cost function.
	- Content cost function.
		- Say you use hidden layer **l** to compute content cost.
			- The earlier the layer **l** in the network more it will force to get the similar image to the content image and vice versa.
		- Use pre-trained network (may be VGG).
		- Let a<sup>[l][C]</sup> and a<sup>[l][G]</sup> be the activations of layer **l** for both images.
		- If a<sup>[l][C]</sup> and a<sup>[l][G]</sup> are similar, both images have similar content.
		- Take the element wise difference between the activations of the both images.
		- Apply gradient descent to insentivise your algorithm to make the image as closer to the content as you want.
	- Style cost function.
		- Say you are using layer **l's** activation to measure "style".
			- N<sup>h</sup>, N<sup>w</sup> and N<sup>c</sup> be height , widths and channels.
		- How correlated the activations across different channels are.
		- Style matrix computatoin.
- Convolution in 1D, 2D and 3D.


## Day 32 | May 2 2020 | Saturday
Today I continued the deep learning specialization part 4 for convolutional neural networks.
- While object detection your model need to detect one object only once.
- Non-max Suppression example.
	- Use IOU (intersection over union) to check the neighboring windows if they are overlapping with the same window which has highest probability of detection or not, if that's the case it will keep the highest probability region and discard others.
- Anchor boxes.
	- Detecting multiple objects in the image.
	- Using multiple anchor boxes to detect objects.
- YOLO (you only look once) algorithm.
	- Sliding windows for the image.
	- Apply Anchor boxing.
	- Apply Apply Non-max Suppression.
- Region Proposals.
	- Propose a regions where there might be an object to detect.
	- Run classification only on that region.
	- Its quite slow.
- Face verification vs. face recognition.
	- Verification.
		- Input Image, name/ID.
		- Output whether the input image is that of the claimed person.
	- Recognition.
		- Has a dataset of K persons.
		- Get an input image.
		- Output ID if the image is any of the K persons (or "Not recognized").


## Day 31 | May 1 2020 | Friday
Today I continued with deep learning specializatino part 4 for convolutional neural networks, and learnt about the object detection.
- Object localization.
	- Ouputing not just the object class, object presence but also its x,y and height and width.
	- The label **y** for the images would also be containing this information (like if there is an object, object class and the boundary parameters).
	- Loss function.
- Landmark detection.
	- Giving labels where you also specify the landmark points (x,y).
		- You can have as many points for land mark as you want.
- Object detectoin.
	- Sliding windows detection.
	- Train network on tightly cropped images of the object.
	- Take a window and slide it over any input image (not tightly cropped) and pass those window cropped part of the image to the network to predict whether the object is present or not.
	- Multiple slidings with different sizes of the window.
	- Computational cost for this operation is high.
- Convolutional implementation of sliding windows.
- Intersection over union (IoU).
	- Area of the overlaped boxes divided by the total area coverd by the boxes.


## Day 30 | April 30 2020 | Thursday
Today I continued with deep learning specialization part 4 for convolutional neural networks,explored the architectures of the CNNs and learnt following topics.
- Neural networks architectures.
	- LeNet - 5 architecture.
	- AlexNet architecture.
	- VGG -16 architecture.
- Very very deep neural networks are difficult to train due to the fact of exploding/vanishing of the gradient.
- Residual Networks (ResNet).
	- Skip connections.
- Network in Network.
	- One by One convolution.
- Inception model.
	- Using multiple filters with different sizes on a single convolutional layer along with pooling filters keeping the output dimension same for all the filters.
	- Let the network decide which filter will be best to use.
	- Requires very high computational cost.
	- Bunch of inception modules are combined to create a single Inception model.
- Open source implimentation.
- [Transfer learning.](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a)
- Data Augmentation.
	- Artificially creating the data.
	- Flipping, RGB variation and cropping.
- State of computer vision. 


## Day 29 | April 29 2020 | Wednesday
Today I started the deep learning specialization part 4 for convolutional neural networks, and learnt the following things.
- Motvaion for computer vision.
- Convolution operation & Edge detection.
- If you convolve an image of N<sup>2</sup> with a filter of f<sup>2</sup> the output image size would be (N-f+1)<sup>2</sup>. It shrinks the output image.
- Padding is used to overcome shrinking of the image to avoid image to become too small if your network is too deep. 
	- p = (f-1)/2 where f is usually odd.
- Strided convolution.
	- Image size after strided convolution: floor((n+2p-f)/s + 1) where s = stride size.
- Convolution on volumes (3D (RGB) images).
	- Filter for this convolutional will also be 3D.
	- Output will be a R<sup>2</sup> image.
	- We can have as many filters as we want or need.
- One layer of convolutional neural network.
- Simple Convolutional network example from feature learning to predictions.
	- Convolutional Layer (Feature learning)
	- Flatten (fully connected) layers.
	- Output layer.
- Pooling layer.
	- Max pooling, Min Pooling and Average pooling.
	- Has hyperparameters to tune, but no learnable parameters (through gradient descent).
- Example of convolutional neural network LeNet.
- Why using Convolutional neural networks.
	- Parameter sharing.
		- A feature detector that's useful in one part of the image is probably useful in another part of the image.
	- Sparsity of connections.
		- In each layer, each output value depends only on a small number of inputs.


## Day 28 | April 28 2020 | Tuesday
Today I completed the deep learning specialization part 3.
- Addressing the data missmatch.
	- Try to understand the differences between the training and the dev, test set. 
	- Artificial data synthesis.
	- Problem of sythesis only the subset of the data. (for example creating only a few types of cars and not all type of car images)
- Transfer learning.
	- pre training and fine tunning.
	- Transfer learning from A ---> B.
		- Apply transfer learning when problem A and B have the same type of input X.
		- You have lot more data for problem A than problem B.
		- Low level features learnt from problem A can be very helpful for problem B.
- Multitask learning.
	- When you have same problem. (e.g reconizing sign boards, cars and roads)
- End to End Deep learning.
	- Avoiding pipelines for different steps and performing input to output map in a single neural network.
		- You need to have a lot of data.
- Whether to use end to end deep learning.
	- Need a huge amount of data.


## Day 27 | April 27 2020 | Monday
Today I continued with the deep learning specialization part 3.
- Error Analysis.
	- Manually examining the mistakes made by the model.
	- Checking the labels of the miss-classified samples (small sample) manually to confirm if the labels are correct. 
	- Preparing spread sheet for better analysis.
- Cleaning up the incorrectly labelled data.
	- Small number of hours of manually examination can help you figure out the real problem and can solve the problem much quickly than randomly changing paramters and algorithms.
- Build your first machine learning system quickly and then iterate over it again and again to improve.
- Training and testing on different distributions.	
	- Randomly shuffle the distributions of the sets. 
- Bias and Variance with miss-matched data.
	- Train, train-dev set, dev set and test set.
	- Data miss-match 
	- Variance & bias problem.


## Day 26 | April 26 2020 | Sunday
Today I started the deep learning specialization part 3 for structuring the machine learning projects. 
- Orthogonalization in machine learning model improvement.
	- Using specific set of hyperparamters to tune the model performance.
- Single value model evalueation, F1 score and Accuracy.
	- Precision and Recall.
	- False Positives and False Negatives.
	- True Positives and True Negatives.
- Satisficing and Optimizing metrics.
	- if you have N metrices optimize one as much as you can by keeping N-1 as satisficing as possible.
- Train, Validation and Test.
	- Keep the distribution same.
	- Random shuffle of data.
	- Size of the train, validation and test sets.
- Evaluation of metrics also depends upon the user exceptance and company vision and needs.
- You can even customize you error function for a perticular class to be predicted or not predicted by either multiplying with a constant.
- Human level performance and even better.
	- Avoidable bias.
	- Understanding human level error/performance.
		- Team of expert radiologists classifying images.
	- Variance also needs to be reduced while achieving or surpassing human level performance.
- Improving model performance.


## Day 25 | April 25 2020 | Saturday
Today I completed the deep learning specialization part 2.
- Grid Search.
- Random selection of parameters (Sampling randomly).
- Exploitation of the parameter space where the results are better.
- Using an appropriate scale for random selection.
	- Example for tuning alpha
		1. r = -x * np.random.rand()  where x < 0
		2. alpha = 10<sup>r</sup>
	- Example for tuning mumentom.
		1. r = [-x,-y] where -x < -y < 0
		2. beta = 1 - 10<sup>r</sup>
- Expert knowledge via reading research papers and experimentations.
- Babysitting approach for tunning hyperparameters.
	- Keep monitering the learning curve (or loss curve) and change parameters accordingly.
- Training multiple models in parallel with different hyperparameter optimization.
- Batch normalization.
	- Makes hyperparameter search problem much easier.
	- Essentially it makes the neural networks much more robust against the choice of hyperparamters.
	- Can work very well even in a very big range of paramters.
	- Easily train deep neural networks.
	- Batch normalization normalizes the data comming from the hidden layers and going into the hidden layers throughout the network.
		- Normalizing the *Z<sup>[l]</sup>* where *Z<sup>[l]</sup> = W<sup>[l]</sup>y<sup>'[l-1]</sup>*
		- For normalization you can choose any distribution that fits your problem.
	- Applied with mini-batches.
	- Understanding of why does batch normalization works.
	- Make weights in the later layers (say 10<sup>10th</sup>) more robust to changes than earlier layers.
	- If you learn a mapping of **X --> y** if distribution of **X** changes then the model might need to be re-trained.
	- Batch Normalization actually reduces the problem of distribution among the layers.
	- Batch Normalization at Test Time.
		- Use estimated exponentially weighted average (across mini-batches)
- Softmax Regression.
	- Choose the close with maximum probability for more than 2 class classification problem.
	- Training with softmax classifier.
- Local optima problems in high dimentional space.
- Introduction to tensorflow deeplearning framework.


## Day 24 | April 24 2020 | Friday
Today I started with Gradient checking and moved on to learn new topics on hyperparameter optimization.
- Gradient checking.
	- Checking the value of original gradient against gradient generated after adding nudges to the weights. 
	- Gradient checking does not work with Dropout.
- Optimization Algorithms.
	- Mini-batch Gradient Descent.
	- Mini-batch Gradient Descent Implimentation.
	- Batch vs Mini-batch Gradient Descent (curve for loss drop).
		- For stochastic Gradient Descent keep the mini-batch 1.
	- [Important Optimization of Gradient Descent](https://ruder.io/optimizing-gradient-descent/)
	- Exponentially weighted averages.
		- V<sub>t</sub> = *B*V<sub>t-1</sub> + (1-*B*)x<sub>t</sub> :. *B = 0.9*
	- Exponentially weighted averages Implimentation. 
	- Gradient Descent with momentum.
		- Almost always faster than simple Gradient Descent.
	- RMSprop.
		- Controlls the learning rate in different directions according to the need.
	- Adim (Adaptive moment estimation) Optimization Algorithm.
		- Works fine for a large range of deep learning algorithms.
		- Combine both momentum and RMSprops.
	- Learning Rate Decay.
		- Implimentation of Learning Rate Decay.
		- *alpha = (1/(1+decay_rate * epoch_number)) * alpha*
		- *a = small_constant<sup>epoch_number</sup>* small_constant < 1.


## Day 23 | April 23 2020 | Thursday
Today I started learning more about hyperparamter optimization for neural networks.
- Regularization.
	- L1 & L2 regularization.
- Regularization penalizes the weights to reduces the overfitting.
- Dropout Regularization.
	- Randomly shutdown some of the units in the network.
	- For larger layers use high value for dropout (since larger layers have larger weight matrix).
- Dropout implimentation.
- Why does Dropout Work.
	- Can't rely on any one feature, so have to spread out weights.
- Other regularizations.
	- Data augmentation.
	- Early Stopping.
- Input Normalization for neural networks.
	- Bring all features to a common scale.
- Vanishing/Exploding Gradient.
	- The slope of the functions either get exponentially small or exponentially large.
- Solution to Vanishing/Exploding Gradient.
	- Random initialization.
	- The larger the number of neurons the smaller the values of connected weights should be.
	- Example random initialization: W<sup>[l]</sup> = np.random.randn(shape) * np.sqrt(1/n<sup>[l-1]</sup>)
- Numerical Approximations of Gradients.


## Day 22 | April 22 2020 | Wednesday
Today I completed the deep learning specialization part 1 and went on for the second part. The progress for today is given below.
- Building blocks of a neural network.
- Forward and Backward propogation.
- parameters and hyperparameters.
	- hyperparameters.
		- Learning rate.
		- Activation Function.
		- Number of layers.
		- Number of layer to choose.
		- Loss function.
	- parameters.
		- weights & biasis.
- Finding suitable hyperparameters is an iterative process.
- Train, Validation and Test set split.
- Generally train = 60%, validation = 20% and test = 20%.
	- Distribution of the data across these folds matters alot.
- When using Big Data (for example 1,000,000 rows).
	- You can take the validation/test set even smaller than 20% (for example 10,000).
	- The idea for validation/test set is to find out if the model is performing better or not.
	- Sometimes this split can be train = 99.5, validation = 0.25 and test = 0.25, when you have 10s of 100s of millions of samples.
	- Destribution of the data matters alot in these splits.
- Bias and Variance trade off.
- Basic machine learning recipe. 
	- Try to figure out problems in your model.
	- Fix them one by one.
	- Iterative process. 


## Day 21 | April 21 2020 | Tuesday
Today I started derivatives of activation functions, backpropogation for neural networks.
- Derivatives of the activation functions.
- Gradient descent and Backpropogation implimentation details.
- Random initialization.
- Deep neural network.
- Forward pass for deep neural networks.
- Dimensions of matrices for deep neural networks.
- Motivation why deep representation of neural networks.
	- With shallow neural networks we would required to have exponential units.


## Day 20 | April 20 2020 | Monday
Today I kept going with the deep learning specialization part one and learned following things.
- Neural network overview.
- 2 layers neural network.
- Implimentation instructions for neural networks.
- Explaination of vectorized implimentation.
- Activation Functions.
	- Sigmoid
	- ReLU
	- tanh
	- leaky-ReLU
	- softmax


## Day 19 | April 19 2020 | Sunday
Today I started the deep learning specialization, and started the first course.
- Course overview and motivation for deeplearning.
- Simple neural network example.
- Supervised learning and types of neural networks.
- Learning curve of deep neural network with respect to the amount of data (and comparision with other ML algorithms).
- Feature vectors of data (images). 
- Sigmoid function and bias.
- Logistic regression cost function.
	- *Cost(y, y<sup>'</sup> ) = - ( (y)log(y<sup>'</sup>) + (1-y)(log(1-y<sup>'</sup>)))*
	- Interestingly one of the two term gets 0 depending upon the *y* values (0,1).
- Gradient Descent.
- Intuition of derivatives.
	- Slope of the function with respect to some given variable *x*.
- Computation Graphs.
	- Computation Graph example.
- Logistic Regression Gradient Descent.
- Vectorization and explicite for loops.
- Logistic Regression and Gradient Descent vectorization.	


## Day 18 | April 18 2020 | Saturday
Today I started learning the Eigen Decomposition, Singular Value Decomposition.
- Eigen Decomposition.
	- Defined for only squared matrices.
	- Eigen Values & Eigen vectors.
	- For *NxN* matrix there are *N* eigen values and *N* eigen vectors.
	- If **Av =** *y***v** given that **A** is a matrix, **v** is a vector and **y** is a scaler. Then **v** is an Eigen vector for matrix **A** and *y* is the eigen value.
	- Eigen Decomposition process.
		- Take the diagonal elements and subtract *Y* assuming *Y* to be an eigen value.
		- Take determinent of the matrix.
		- Find all the possible values of *Y* (which are going to be generally *N* given that matrix is *NxN*).
		- In the initial step replace all the appearances of *Y* with one eigen value at a time and find a vector which is in the Null space of the matrix ignoring the trivial case.
		- This vector is the eigen vector.
	- Diagonalization.
- Singular Value Decomposition.
	- Conceptually similar to Eigen decomposition but it can work with rectangular matrices as well, by multiplying a matrix with its transpose. 
	


## Day 17 | April 17 2020 | Friday
Today I started learning about the matrix inverses, orthogonalities & Projections and Matrix least square for model fitting.
- Matrix Inverse equation.
	1. **Ax = b**
	2. **A<sup>-1</sup>Ax = A<sup>-1</sup>b**
	3. **Ix = A<sup>-1</sup>b**
	4. **x = A<sup>-1</sup>b**
	- Matrix inverse is side-dependent.
- Matrix inverse using row reduction echelon form.
	- Agumentation of Identity matrix with the original matrix **(A | I)**
	- Converting the original matrix to identity matrix using linear operation which will broadcast along to Identity matrix as well.
	- At the end if you get an identity matrix for the original matrix (on left) you have the inverse matrix of the original matrix (on the right size).
	- If you get zeros in any row of the original matrix during this process then the solution does not exist.
	- **(A|I) ==> (I|A<sup>-1</sup>)**
- Projections and orthogonalization.
	- Projection of point *b* on a vector **a** with some scaler *B* in R<sup>2</sup> is given as.
		1. **a<sup>T</sup>** (b-**a**B) = 0
		2. **a<sup>T</sup>** b - **a<sup>T</sup>a** B = 0
		3. **a<sup>T</sup>a** B = **a<sup>T</sup>** b
		4. B = **a<sup>T</sup>** b / **(a<sup>T</sup>a)**
	- Projection of point *b* on a matrix **A** with some vector **x** in R<sup>N</sup> is given as.
		1. **A<sup>T</sup>**(b - **Ax**) = **O**
		2. **A<sup>T</sup>** b - **A<sup>T</sup>Ax** = **O**
		3. **A<sup>T</sup>Ax** = **A<sup>T</sup>** b
		4. **(A<sup>T</sup>A)<sup>-1</sup> (A<sup>T</sup>A)x = (A<sup>T</sup>A)<sup>-1</sup>A<sup>T</sup>** b
		5. **x = (A<sup>T</sup>A)<sup>-1</sup>A<sup>T</sup>** b
		6. **x = A<sup>-1</sup>A<sup>-T</sup>A<sup>T</sup>** b
		7. **x = A<sup>-1</sup>** b
	- Orthogonal & parallel vector components.
	- Orthogonal Matrices.
	- Gram-Schimdt and QR decomposition.
- Least squares for model fittings.
	- Fixed parameters.
		- Parameters that you set for the model.
	- Free parameters.
		- Parameters that the model learns from the data.
	- Finding coefficients of linear models.
	- **Ax = y** ideally **y** needs to be in the column space of **A**. But typically its not.
	- So we can write the above equation as **Ax + e = y** where **e** is the error vector.


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
	- **A<sup>T</sup>** has the same column space as **A**


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