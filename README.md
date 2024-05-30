# Multivariate Polynomial Regression Model

## Description

This is a multivariate polynomial regression model written in Python and utilizing NumPy that I wrote after learning about the basics of Machine Learning.

## The Details

### A Quick Note About This Section in General

The explanations given here are far too lengthy given the simplicity of the model. I go through the theory and math behind the model and cover it far more "in-depth" than necessary. This section is mainly meant as a way for me to strengthen my understanding of the concepts by explaining them.

### Model Function

Of course, the goal of regression is to fit some line to a set of data. The model function represents that line and is what the model uses to make its predictions. Traditionally, a univariate linear regression model uses the function

``` math
f_{w,b}(x^{(i)}) = wx^{(i)} + b
```

where $`x^{(i)}`$ denotes the $`i^{\text{th}}`$ input value and $`w`$ and $`b`$ denotes parameters of the model, namely the weight and bias repsectively.

For a multivariate linear regression model, there are multiple input features, which we denote as matrix $`\mathbf{X}`$. Thus, vector $`\mathbf{x}^{(i)}`$ denotes the values of the features for the $`i^{\text{th}}`$ training example and vector $`\mathbf{x}_{j}`$ denotes the values for the $`j^{\text{th}}`$ feature across all training examples. Naturally, $`x^{(i)}_{j}`$ denotes the value of the $`j^{\text{th}}`$ feature for the $`i^{\text{th}}`$ training example. Typically, the features matrix $`\mathbf{X}`$ is of size $`m \times n`$, which $`m`$ denotes the number of training examples and $`n`$ denotes the number of features. The function for a multivariate linear regression model would be

``` math
f_{w,b}(\mathbf{x}^{(i)}) = w_{1}x^{(i)}_{1} + w_{2}x^{(i)}_{2}+ \dotsb + w_{n}x^{(i)}_{n} + b
```

which could also be expressed as

``` math
f_{w,b}(\mathbf{x}^{(i)}) = \sum_{j=1}^{n}{w_{j}x^{(i)}_{j}} + b
```

More concisely, if we let vector $`\mathbf{w}`$ denote the weights for the model for each respective feature, then our function can be written as

``` math
f_{w,b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)} + b
```

To accommodate a nonlinear associated between input features and target outputs, we require a polynomial regression model, which can be done by introducing polynomial terms into our model, such as

``` math
f_{w,b}(\mathbf{x}^{(i)}) = w_{1}\sqrt{x^{(i)}_{1}} + w_{2}x^{(i)}_{1} + w_{3}(x^{(i)}_{1})^{2} + \dotsb + b
```

For this model in particular, we choose a degree $`d`$ and let our model be

``` math
\boxed{f_{w,b}(\mathbf{x}^{(i)}) = w_{1}x^{(i)}_{1} + w_{2}(x^{(i)}_{1})^{2} + \dotsb + w_{d}(x^{(i)}_{1})^{d} + w_{d+1}x^{(i)}_{2} + w_{d+2}(x^{(i)}_{2})^{2} + \dotsb + w_{2d}(x^{(i)}_{2})^{d} + \dotsb + w_{(n-1)d+1}x^{(i)}_{n} + w_{(n-1)d+2}(x^{(i)}_{n})^{2} + \dotsb + w_{nd}(x^{(i)}_{n})^{d} + b}
```

Perhaps summation notation is more concise; we can write

``` math
\boxed{f_{w,b}(\mathbf{x}^{(i)}) = \sum_{j=1}^{n}{\sum_{k=1}^{d}{w_{(j-1)d+k}(x^{(i)}_{j})^{k}}} + b}
```

Of course, we could also combine different features together to create a term like $`w_{1}x^{(i)}_{1}x^{(i)}_{2}`$ but the model I chose for this project does not do this, for the sake of simplicity.

### Polynomial Feature Mapping

The main downsides of defining our model function as polynomial such as the ones above is that they complicate our cost function (making its partial derivatives difficult to compute) and that it will take longer to compute the model's prediction (two summations means two `for`-loops which means $`\approx \text{O}(N\log(N))`$ time complexity).

To solve this, map polynomial features to the raw input data and then perform linear regression. For example, if we wanted the feature $`(x^{(i)}_{j})^{3}`$, we would cube all the elements in column $`j`$ of matrix $`\mathbf{X}`$ and then perform linear regression.

For a quick explanation of why this works suppose we are dealing with univariate input data and have the points $`(2, 7)`$, $`(3, 27)`$, and $`(4, 65)`$ representing the feature and target values respectively. We notice that the data possesses a trend roughly matching that of a cubic polynomial. As such, we map the input features (by cubing them) to get the points $`(8, 7)`$, $`(27, 27)`$, and $`(64, 65)`$. Performing linear regression now yields an accurate model as the points are very close to the line $`y = x`$.

From now on, we assume that our features have been mapped to polynomial ones and we continue to denote the feature matrix as $`\mathbf{X}`$. Since we are now only concerned with linear regression, our model function can be rewritten as:

``` math
\boxed{f_{w,b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)} + b}
```

### Data Normalization

The goal of normalization is to get all input features within a similar range, typically one centered around $`0`$ and within a small interval like $`[-1, 1]`$. This is very useful during the gradient descent process as it ensures each iteration changes each parameter (about) equally  and increases the change of convergence to a minima.

There are many ways to do this but for this project, I chose z-score normalization. This is done by reassinging each $`x^{(i)}_{j}`$ to

``` math
x^{(i)}_{j} := \frac{x^{(i)}_{j} - \mu_{j}}{\sigma_{j}}
```
