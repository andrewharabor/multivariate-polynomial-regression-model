# Multivariate Polynomial Regression Model

## Description

This is a multivariate polynomial regression model written in Python and utilizing NumPy that I wrote after learning about the basics of Machine Learning.

## The Details

### Model Function

Traditionally, a univariate linear regression model uses the function $$f\_{w,b}(x^{(i)}) = wx^{(i)} + b$$ where $x^{(i)}$ denotes the $i^{\text{th}}$ input value and $w$ and $b$ denotes parameters of the model, namely the weight and bias repsectively.

For a multivariate linear regression model, there are multiple input features, which we denote as matrix $\mathbf{X}$. Thus, vector $\mathbf{x}^{(i)}$ denotes the values of the features for the $i^{\text{th}}$ training example and vector $\mathbf{x}\_{j}$ denotes the values for the $j^{\text{th}}$ feature across all training examples. Naturally, $x^{(i)}\_{j}$ denotes the value of the $j^{\text{th}}$ feature for the $i^{\text{th}}$ training example. Typically, the features matrix $\mathbf{X}$ is of size $m \times n$, which $m$ denotes the number of training examples and $n$ denotes the number of features. The function for a multivariate linear regression model would be $$f\_{w,b}(\mathbf{x}^{(i)}) = w\_{1}x^{(i)}\_{1} + w\_{2}x^{(i)}\_{2}+ \dotsb + w\_{n}x^{(i)}\_{n} + b$$ which could also be expressed as $$f\_{w,b}(\mathbf{x}^{(i)}) = \sum\_{j=1}^{n}{w\_{j}x^{(i)}\_{j}} + b$$ More concisely, if we let vector $\mathbf{w}$ denote the weights for the model for each respective feature, then our function can be written as $$f\_{w,b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)} + b$$

To accommodate a nonlinear associated between input features and target outputs, we require a polynomial regression model, which can be done by introducing polynomial terms into our model, such as $$f\_{w,b}(\mathbf{x}^{(i)}) = w\_{1}\sqrt{x^{(i)}\_{1}} + w\_{2}x^{(i)}\_{1} + w\_{3}(x^{(i)}\_{1})^{2} + \dotsb + b$$ For this model in particular, we chose a degree $d$ and let our model be $$\boxed{f\_{w,b}(\mathbf{x}^{(i)}) = w\_{1}x^{(i)}\_{1} + w\_{2}(x^{(i)}\_{1})^{2} + \dotsb + w\_{d}(x^{(i)}\_{1})^{d} \\ + w\_{d+1}x^{(i)}\_{2} + w\_{d+2}(x^{(i)}\_{2})^{2} + \dotsb + w\_{2d}(x^{(i)}\_{2})^{d} \\ + \dotsb \\ + w\_{(n-1)d+1}x^{(i)}\_{n} + w\_{(n-1)d+2}(x^{(i)}\_{n})^{2} + \dotsb + w\_{nd}(x^{(i)}\_{n})^{d} \\ + b}$$ Perhaps summation notation is more concise; we can write $$\boxed{f\_{w,b}(\mathbf{x}^{(i)}) = \sum\_{j=1}^{n}{\sum\_{k=1}^{d}{w\_{(j-1)d+k}(x^{(i)}\_{j})^{k}}} + b}$$ Of course, we could also combine different features together to create a term like $w\_{1}x^{(i)}\_{1}x^{(i)}\_{2}$ but the model I chose for this project does not do this, for the sake of simplicity.

### Polynomial Feature Mapping

The main downsides of defining our model function as polynomial such as the ones above is that they complicate our cost function (making its partial derivatives difficult to compute) and that it will take longer to compute the model's prediction (two summations means two `for`-loops which means $\approx \text{O}(N\log(N))$ time complexity).
