# Multivariate Polynomial Regression Model

## Description

This is a multivariate polynomial regression model written in Python and utilizing NumPy that I wrote after learning about the basics of Machine Learning.

## The Code

~~The code for this project is self-documented and basic knowledge of Machine Learning should suffice in order to understand it.~~

_Correction:_ It turns out that vectorizing everything convoluted the code. The formulas should match those presented in the following section but thorough knowledge of linear algebra and intuition for Machine Learning is required to understand what is actually happening. The code is documented in a way that explains the high-level idea behind the functions but not the details about the computations they carry out. I tried to make things as readable as possible (i.e., avoid outright one-liners) but there's not much else I can do to document a complicated linear algebra formula.

On the upside, after vectorizing everything, the code runs _way_ faster. For example, 100,000 iterations of gradient descent used to take a couple minutes but after vectorization, it typically runs in under 10 seconds.

## The Theory and Math

### A Brief Note About This Section

This section is mainly meant as a way for me to strengthen my understanding of the concepts by explaining them. The explanations given here go into the theory and math behind the model and are far more lengthy than needed.

### Model Function

Of course, the goal of regression is to fit some line to a set of data. The model function represents that line and is what the model uses to make its predictions. Traditionally, a univariate linear regression model uses the function

``` math
f_{w,b}(x^{(i)}) = wx^{(i)} + b
```

where $`x^{(i)}`$ denotes the $`i^{\text{th}}`$ input value and $`w`$ and $`b`$ denotes parameters of the model, namely the weight and bias repsectively.

For a multivariate linear regression model, there are multiple input features, which we denote as matrix $`\mathbf{X}`$. Thus, vector $`\mathbf{x}^{(i)}`$ denotes the values of the features for the $`i^{\text{th}}`$ training example and vector $`\mathbf{x}_{j}`$ denotes the values for the $`j^{\text{th}}`$ feature across all training examples. (Note that all vectors are assumed to be column vectors.) Naturally, $`x^{(i)}_{j}`$ denotes the value of the $`j^{\text{th}}`$ feature for the $`i^{\text{th}}`$ training example. Typically, the features matrix $`\mathbf{X}`$ is typically in $`\mathbb{R}^{m \times n}`$, where $`m`$ denotes the number of training examples and $`n`$ denotes the number of features. The function for a multivariate linear regression model would be

``` math
f_{w,b}(\mathbf{x}^{(i)}) = w_{1}x^{(i)}_{1} + w_{2}x^{(i)}_{2}+ \dotsb + w_{n}x^{(i)}_{n} + b
```

which could also be expressed as

``` math
f_{w,b}(\mathbf{x}^{(i)}) = \sum_{j=1}^{n}{w_{j}x^{(i)}_{j}} + b
```

More concisely, if we let vector $`\mathbf{w}`$ denote the weights for the model for each respective feature, then our function can be written using the dot product.

``` math
f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w}^{\text{T}}\mathbf{x}^{(i)} + b
```

To accommodate a nonlinear association between input features and target outputs, we require a polynomial regression model, which can be done by introducing polynomial terms into our model, such as

``` math
f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = w_{1}\sqrt{x^{(i)}_{1}} + w_{2}x^{(i)}_{1} + w_{3}(x^{(i)}_{1})^{2} + \dotsb + b
```

For this model in particular, we choose a degree $`d`$ and let our model be

``` math
f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = w_{1}x^{(i)}_{1} + w_{2}(x^{(i)}_{1})^{2} + \dotsb + w_{d}(x^{(i)}_{1})^{d} + w_{d+1}x^{(i)}_{2} + w_{d+2}(x^{(i)}_{2})^{2} + \dotsb + w_{2d}(x^{(i)}_{2})^{d} + \dotsb + w_{(n-1)d+1}x^{(i)}_{n} + w_{(n-1)d+2}(x^{(i)}_{n})^{2} + \dotsb + w_{nd}(x^{(i)}_{n})^{d} + b
```

Perhaps summation notation is more concise; we can write

``` math
f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \sum_{j=1}^{n}{\sum_{k=1}^{d}{w_{(j-1)d+k}(x^{(i)}_{j})^{k}}} + b
```

(I know this model function is probably terribly confusing; I try to explain it better in the next subsection.) Of course, we could also combine different features together to create a term like $`w_{1}x^{(i)}_{1}x^{(i)}_{2}`$ but the model I chose for this project does not do this, for the sake of simplicity.

### Polynomial Feature Mapping

The main downside of defining our model function as polynomial such as the ones above is that it complicates our cost function, making the gradients difficult to compute and limiting the amount of vectorization we can perform. To solve this, map polynomial features to the raw input data and then perform linear regression. For example, if we wanted the feature $`(x^{(i)}_{j})^{3}`$, we would cube all the elements in column $`j`$ of matrix $`\mathbf{X}`$ and then perform linear regression.

For a quick explanation of why this works suppose we are dealing with univariate input data and have the points $`(2, 7)`$, $`(3, 27)`$, and $`(4, 65)`$ representing the feature and target values respectively. We notice that the data possesses a trend roughly matching that of a cubic polynomial. As such, we map the input features (by cubing them) to get the points $`(8, 7)`$, $`(27, 27)`$, and $`(64, 65)`$. Performing linear regression now yields an accurate model as the points are very close to the line $`y = x`$.

From now on, we assume that our features have been mapped to polynomial ones and we continue to denote the features matrix as $`\mathbf{X}`$ and the number of features as $`n`$. As covered in the previous subsection, our new features matrix will consist of raising the elements old features matrix to every power between $`1`$ and $`d`$ inclusive and concatenating the resulting matrices together. (To clear up any confusion, this is done column-wise in the code in `model.py` but both yield the same result.) Mathematically, we can express this reassignment as

``` math
\boxed{\mathbf{X} := \mathbf{X} \vert \mathbf{X}^{\circ 2} \vert \mathbf{X}^{\circ 3} \vert \dotsb \vert \mathbf{X}^{\circ d}}
```

where $`\mathbf{A} \vert \mathbf{B}`$ denotes augmenting (concatenating column-wise) the matrices $`\mathbf{A}`$ and $`\mathbf{B}`$ while $`\mathbf{A}^{\circ k}`$ denotes the matrix $`\mathbf{A}`$ raised to the Hadamard (element-wise) power of $`k`$. Note that here I use the definition operator $`:=`$ to denote a statement of assignment as opposed to a statement of equality (much like `=` versus `==` in code). If any of this caused more confusion about polynomial feature mapping or the model function for this project, I apologize; hopefully the code makes more sense.

Since we are now only concerned with linear regression, our model function can be rewritten as

``` math
\boxed{f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w}^{\text{T}}\mathbf{x}^{(i)} + b}
```

Alternatively, we could have our function output a vector of predictions given the entire data set if we define it as

``` math
\boxed{f_{\mathbf{w},b}(\mathbf{X}) = \mathbf{X}\mathbf{w} + b\mathbf{1}}
```

where $`\mathbf{1}`$ denotes the ones vector of appropriate size (in this case $`\mathbb{R}^{m \times 1} `$). We distinguish between these two definitions of our model function based on whether it is a function of a vector or a matrix.

### Data Normalization

The goal of normalization is to get all input features within a similar range, typically one centered around $`0`$ and within a small interval like $`[-1, 1]`$. This is very useful during the gradient descent process as it ensures each iteration changes every parameter (about) equally  and increases the chance of convergence to a minima.

There are many ways to perform normalization but for this project, I chose z-score normalization. This is done by reassigning each $`x^{(i)}_{j}`$ to

``` math
\boxed{x^{(i)}_{j} := \frac{x^{(i)}_{j} - \mu_{j}}{\sigma_{j}}}
```

Here, $`\mu_{j}`$ denotes the mean of vector $`\mathbf{x}_{j}`$ and $`\sigma_{j}`$ denotes the standard deviation of vector $`\mathbf{x}_{j}`$.

### Loss and Cost Functions

With the data preprocessing out of the way, we can move on to the core of the regression algorithm.

Just like with the data normalization function and other aspects of ML, there are a wide variety of cost functions to choose from, each with their own benefits. For the sake of simplicity however, I chose the mean squared error (MSE) cost function. It also has the very useful property that for regression problems, it is always convex, which means there is only one minima for gradient descent to converge to.

For the MSE cost function, we define the loss function as the error between predicted and expected values for a single example. Its formula is given by

``` math
L(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = \frac{1}{2}(f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^{2}
```

where $`y^{(i)}`$ denotes the actual/expected target value for the $`i^{\text{th}}`$ training example. Squaring the difference between $`f_{\mathbf{w},b}(\mathbf{x}^{(i)})`$ and $`y^{(i)}`$ ensures underestimates are punished just as much as overestimates while also placing greater emphasis on larger residuals.

However, before moving on to

The cost function is defined as the average of the loss function across all examples. It is given by

``` math
J(\mathbf{w}, b) = \frac{1}{m}\sum_{i=1}^{m}{L(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})}
```

If we expand using the definition of the loss function and the definition of our model function, the cost function can be expressed as

``` math
J(\mathbf{w}, b) = \frac{1}{2m}\sum_{i=1}^{m}{(\mathbf{w}^{\text{T}}\mathbf{x}^{(i)} + b - y^{(i)})^{2}}
```

By using the definition $`f_{\mathbf{w},b}(\mathbf{X})`$ of our model function, we can vectorize the cost function as

``` math
J(\mathbf{w}, b) = \frac{1}{2m}\Vert \mathbf{X}\mathbf{w} + b\mathbf{1} - \mathbf{y}\Vert ^{2}
```

where vector $`\mathbf{y}`$ denotes the expected target values and $`\Vert \mathbf{v}\Vert `$ denotes the L2 norm of vector $`\mathbf{v}`$. The vectorized cost function is much faster to compute by using NumPy but is pretty abusive of notation and therefore harder to understand (at least in my opinion). Nevertheless, we will continue using the vectorized implementation.

### Overfitting and Regularization

While the cost function defined above may work for simpler cases of multivariate regression, it is susceptible to overfitting. This is where the the model function is over-optimized to fit the training data. Especially with a higher degree polynomial, the cost on the training set may be minimized but it will likely not match more general trends in the data, meaning the model will perform poorly on any new examples it is tested on. While overfitting can be solved by using more training examples, this may not always be an option, hence the need for regularization.

In order to mitigate overfitting, the model should punished for having high values of the parameters in vector $`\mathbf{w}`$. This will reduce variance and result in a smoother curve described by the model function. We can implement this by adding the following regularization term to our cost function

``` math
\frac{\lambda}{2m}\sum_{j=1}^{n}{w_{j}^{2}} = \frac{\lambda}{2m}{\Vert \mathbf{w}\Vert }^{2}
```

where $`\lambda`$ denotes the regularization parameter (with $`\lambda > 0`$) that determines how much the model should be punished for higher weights. Just how the residuals are squared in the cost function, each weight is squared for the similar reasons. Note that the bias $`b`$ is not usually regularized. The updated and final cost function is then

``` math
\boxed{J(\mathbf{w}, b) = \frac{1}{2m}(\Vert \mathbf{X}\mathbf{w} + b\mathbf{1} - \mathbf{y}\Vert ^{2} + \lambda\Vert \mathbf{w}\Vert ^{2})}
```

### Gradient Descent

The purpose of gradient descent is to "descend" down the cost function to a local/global minima in order to find the optimal parameters for the model. Mathematically, this can be expressed as

``` math
\boxed{\min_{\mathbf{w},b}{J(\mathbf{w},b)}}
```

In order to this, we need the gradient of the cost function, which reveals the "direction" in which the function increases most steeply. By "going" in the opposite direction, we "descend" down the cost function. To compute the gradient, we need the partial derivatives of the cost function with respect to its parameters. The partial derivatives are given by

``` math
\frac{\partial J}{\partial w_{j}} = \frac{1}{m}\sum_{i=1}^{m}{(\mathbf{w}^{\text{T}}\mathbf{x}^{(i)} + b - y^{(i)})x_{j}^{(i)}} + \frac{\lambda}{m}w_{j}
```

and

``` math
\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}{(\mathbf{w}^{\text{T}}\mathbf{x}^{(i)} + b - y^{(i)})}
```

Of course, we can express these in a vectorized manner if we use $`f_{\mathbf{w},b}(\mathbf{X})`$ in combination with a bit of linear algebra.

``` math
\boxed{\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m}(\mathbf{X}^{\text{T}}(\mathbf{X}\mathbf{w} + b\mathbf{1} - \mathbf{y}) + \lambda \mathbf{w})}
```

``` math
\boxed{\frac{\partial J}{\partial b} = \frac{1}{m}\mathbf{1}^{\text{T}}(\mathbf{X}\mathbf{w} + b\mathbf{1} - \mathbf{y})}
```

Now that we have the partial derivatives, we can move on to the actual gradient descent process itself. This involves tens or hundreds of thousands of iterations with the following updates to the model's parameters at each step

``` math
\mathbf{w} := \mathbf{w} - \alpha \frac{\partial J}{\partial \mathbf{w}}
```

``` math
b := b - \alpha \frac{\partial J}{\partial b}
```

It is important to note that the update steps should be performed simultaneously, that is the partial derivatives should both be computed before updating the respective parameters. More explicitly, the update steps can be expressed as

``` math
\boxed{\mathbf{w} := \mathbf{w} - \frac{\alpha}{m}(\mathbf{X}^{\text{T}}(\mathbf{X}\mathbf{w} + b\mathbf{1} - \mathbf{y}) + \lambda \mathbf{w})}
```

``` math
\boxed{b := b - \frac{\alpha}{m}\mathbf{1}^{\text{T}}(\mathbf{X}\mathbf{w} + b\mathbf{1} - \mathbf{y})}
```

where $`\alpha`$ denotes the learning rate, which determines how fast the algorithm descends to a minima. While a small learning rate will cause the descent to proceed very slowly, a large learning rate will result in the algorithm failing to converge. For normalized data, typically $`0 < \alpha \le 1`$.

### Model Evaluation

With the inner workings of the model dissected, we are now concerned with evaluating our model.

For this project in particular, I used the popular Boston House Prices Dataset which contains about 500 examples in total. I decided to split the data up such that 80% of it is used for training the model to find the optimal parameters while the other 20% is used for testing the model and seeing how well it might perform on new data. To quantify its performance, I settled on the very simple metric of mean absolute percent error (MAPE), which is given by the formula

``` math
\frac{1}{m}\sum_{i=1}^{m}{\bigg\vert\frac{f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}}{y^{(i)}}\bigg\vert}
```

While above is the traditional (statistics) formula for MAPE, we can vectorize it in the spirit of this project as

``` math
\boxed{\frac{1}{m} \Vert (\mathbf{X}\mathbf{w} + b\mathbf{1} - \mathbf{y}) \oslash \mathbf{y} \Vert_{1}}
```

where $`\Vert \mathbf{v} \Vert_{1}`$ denotes the Taxicab/Manhattan/L1 norm of vector $`\mathbf{v}`$ and $`\mathbf{v} \oslash \mathbf{u}`$ denotes Hadamard (element-wise) division between the vectors $`\mathbf{v}`$ and $`\mathbf{u}`$.

After running the program with different parameters (I don't mean the model's parameters $`\mathbf{w}`$ and $`b`$), I eventually found the following "optimal" settings for the model

``` math
d = 3
```

``` math
\lambda = 0.5
```

``` math
\alpha = 0.1
```

for the polynomial degree, regularization parameter, and learning rate respectively. (Granted, I may later change the values in `model.py` without updating this `README.md` as I continue to tweak the model.)

Overall, I found that the model performed with about 10% to 16% MAPE for both the training and testing data. I would like to get it below 10% but I suspect the variation in the dataset simply renders this impossible. Alternatively, some more advanced regression techniques could potentially solve this problem.

## Final Thoughts

Through this project, I was able to build a solid foundation in the basics of Machine Learning, namely regression problems. It was really helpful to write the code for the model from scratch and tweaking the settings (learning rate, regularization parameter, etc.) helped build intuition for things like convergence to a local minima and overfitting. Vectorizing everything proved to be a fun exercise and allowed me to apply my knowledge of linear alegbra. In addition, it was incredible to see how the code ran much faster when most of the computational "heavy-lifting" was done by NumPy. I hope to continue my journey through ML and this project served as an excellent launchpad for doing so.

## Acknowledgements

[Numpy](https://numpy.org/) - Heavily utilized throughout the project and immensely useful for speeding up linear algebra computations.

[Coursera Machine Learning Specialization Course 1: Supervised Machine Learning: Regression and Classification](https://www.coursera.org/learn/machine-learning?specialization=machine-learning-introduction) - Taught me about regression in Machine Learning and was the main source of inspiration for this project; some of the math and code was adapted from this course.

[Linear Regression (Multivariate). Cost Function. Hypothesis. Gradient](https://www.humanunsupervised.com/post/linear-regression-multivariate-cost-function-hypothesis-gradient) - Inspired the complete vectorization of computations in this project, some math formulas adapted from this webpage.
