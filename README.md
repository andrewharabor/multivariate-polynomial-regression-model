# Multivariate Polynomial Regression Model

## Description

This is a multivariate polynomial regression model written in Python and utilizing NumPy that I wrote after learning about the basics of Machine Learning.

## The Details

### A Brief Note About This Section

This section is mainly meant as a way for me to strengthen my understanding of the concepts by explaining them. The explanations given here go into the theory and math behind the model and are far more lengthy than needed. The code for this project is self-documented and basic knowledge of machine learning should suffice in order to understand it.

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
f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w}^{\mathsf{T}}\mathbf{x}^{(i)} + b
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

Of course, we could also combine different features together to create a term like $`w_{1}x^{(i)}_{1}x^{(i)}_{2}`$ but the model I chose for this project does not do this, for the sake of simplicity.

### Polynomial Feature Mapping

The main downsides of defining our model function as polynomial such as the ones above is that they complicate our cost function (making its partial derivatives difficult to compute) and that it will take longer to compute the model's prediction (two summations means two `for`-loops which means $`\approx \text{O}(N^{2})`$ time complexity).

To solve this, map polynomial features to the raw input data and then perform linear regression. For example, if we wanted the feature $`(x^{(i)}_{j})^{3}`$, we would cube all the elements in column $`j`$ of matrix $`\mathbf{X}`$ and then perform linear regression.

For a quick explanation of why this works suppose we are dealing with univariate input data and have the points $`(2, 7)`$, $`(3, 27)`$, and $`(4, 65)`$ representing the feature and target values respectively. We notice that the data possesses a trend roughly matching that of a cubic polynomial. As such, we map the input features (by cubing them) to get the points $`(8, 7)`$, $`(27, 27)`$, and $`(64, 65)`$. Performing linear regression now yields an accurate model as the points are very close to the line $`y = x`$.

From now on, we assume that our features have been mapped to polynomial ones and we continue to denote the feature matrix as $`\mathbf{X}`$. Since we are now only concerned with linear regression, our model function can be rewritten as

``` math
\boxed{f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w}^{\mathsf{T}}\mathbf{x}^{(i)} + b}
```

Alternatively, we could have our function output a vector of predictions given the entire data set if we define it as

``` math
\boxed{f_{\mathbf{w},b}(\mathbf{X}) = \mathbf{X}\mathbf{w} + b\mathbf{1}}
```

where $`\mathbf{1}`$ denotes the ones vector of appropriate size (in this case $`\mathbb{R}^{m \times 1} `$). We distinguish between these two definitions of our model function based on whether its parameter is a matrix or a vector.

### Data Normalization

The goal of normalization is to get all input features within a similar range, typically one centered around $`0`$ and within a small interval like $`[-1, 1]`$. This is very useful during the gradient descent process as it ensures each iteration changes every parameter (about) equally  and increases the chance of convergence to a minima.

There are many ways to perform normalization but for this project, I chose z-score normalization. This is done by reassigning each $`x^{(i)}_{j}`$ to

``` math
\boxed{x^{(i)}_{j} := \frac{x^{(i)}_{j} - \mu_{j}}{\sigma_{j}}}
```

Here, $`\mu_{j}`$ denotes the mean of vector $`\mathbf{x}_{j}`$ and $`\sigma_{j}`$ denotes the standard deviation of vector $`\mathbf{x}_{j}`$. Note that here I use the definition operator $`:=`$ to denote a statement of assignment as opposed to a statement of equality (much like `=` versus `==` in code).

### Loss and Cost Functions

With the data preprocessing out of the way, we can move on to the core of the regression algorithm.

Just like with the data normalization function and other aspects of ML, there are a wide variety of cost functions to choose from, each with their own benefits. For the sake of simplicity however, I chose the mean squared error cost function. It also has the very useful property that for regression problems, it is always convex, which means there is only one minima for gradient descent to converge to.

For the MSE cost function, we define the loss function as the error between predicted and expected values for a single example. Its formula is given by

``` math
\boxed{L(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = \frac{1}{2}(f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^{2}}
```

where $`y^{(i)}`$ denotes the actual/expected target value for the $`i^{\text{th}}`$ training example. Squaring the difference between $`f_{\mathbf{w},b}(\mathbf{x}^{(i)})`$ and $`y^{(i)}`$ ensures underestimates are punished just as much as overestimates while also placing greater emphasis on larger residuals.

However, before moving on to

The cost function is defined as the average of the loss function across all examples. It is given by

``` math
J(\mathbf{w}, b) = \frac{1}{m}\sum_{i=1}^{m}{L(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})}
```

If we expand using the definition of the loss function and the definition of our model function, the cost function can be expressed as

``` math
J(\mathbf{w}, b) = \frac{1}{2m}\sum_{i=1}^{m}{(\mathbf{w}^{\mathsf{T}}\mathbf{x}^{(i)} + b - y^{(i)})^{2}}
```

By using the definition $`f_{\mathbf{w},b}(\mathbf{X})`$ of our model function, we can vectorize the cost function as

``` math
J(\mathbf{w}, b) = \frac{1}{2m}||\mathbf{X}\mathbf{w} + b\mathbf{1} - \mathbf{y}||^{2}
```

where vector $`\mathbf{y}`$ denotes the expected target values and $`||\mathbf{v}||`$ denotes the L2 norm of vector $`\mathbf{v}`$. The vectorized cost function is much faster to compute by using NumPy but is pretty abusive of notation and therefore harder to understand (at least in my opinion). Nevertheless, we will continue using the vectorized implementation.

### Overfitting and Regularization

While the cost function defined above may work for simpler cases of multivariate regression, it is susceptible to overfitting. This is where the the model function is over-optimized to fit the training data. Especially with a higher degree polynomial, the cost on the training set may be minimized but it will likely not match more general trends in the data, meaning the model will perform poorly on any new examples it is tested on. While overfitting can be solved by using more training examples, this may not always be an option, hence the need for regularization.

In order to mitigate overfitting, the model should punished for having high values of the parameters in vector $`\mathbf{w}`$. This will reduce variance and result in a smoother curve described by the model function. We can implement this by adding the following regularization term to our cost function

``` math
\frac{\lambda}{2m}\sum_{j=1}^{n}{w_{j}^{2}} = \frac{\lambda}{2m}{||\mathbf{w}||}^{2}
```

Where $`\lambda`$ denotes the regularization parameter that determines how much the model should be punished for higher weights. Note that the bias $`b`$ is not usually regularized. Our updated and final cost function is then

``` math
\boxed{J(\mathbf{w}, b) = \frac{1}{2m}(||\mathbf{X}\mathbf{w} + b\mathbf{1} - \mathbf{y}||^{2} + \lambda||\mathbf{w}||^{2})}
```
