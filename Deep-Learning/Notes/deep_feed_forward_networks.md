# Deep FeedForward Networks

+ Also known as **feedforward neural networks**, or **multilayer perceptrons (MLPs)**.
+ A feedforward network defines a mapping $y = f(x; \theta)$ and learns the value of the parameters $\theta$ that result in the best function approximation.
+ **Feedforward** because information flows through the function being evaluated from x, through the intermediate computations used to define f, and finally to the output y.
+ When FeedForward neural networks are extended to include feedback connections, they are called **Recurrent Neural Networks**.
+ FeedForward neural networks are called **networks** because they are represented by composing together many different functions.
+ For example, $f(x) = f^{(3)}(f^{(2)}(f^{(1)}(x)))$. 
  + $f^{(1)}$ is called the **first layer** of the network.
  + $f^{(2)}$ is called the **second layer**, and so on.
+ The overall length of the chain gives the **depth** of the model.
+ The final layer of a feedfoward network is called the **output layer**.
+ Because the training data doesn't show the desired output for each of these layers, they are called **hidden layers**.
+ The strategy of deep learning is to learn $\phi$. In this approach, we have a model $y = f(x; \theta, w) = \phi(x; \theta)^Tw$. We now have parameters $\theta$ that we use to learn $\phi$ from a broad class of functions, and parameters w that map from $\phi(x)$ to the desired output. This is an example of a deep feedforward network, with $\phi$ deﬁning a hidden layer.

## Gradient-Based Learning

+ The difference between linear models and neural networks is that the non-linearity of a neural network causes most interesting loss functions to become [non-convex](https://www.solver.com/convex-optimization#Convex%20Functions).
+ This means that neural networks are usually trained by using iterative, gradient-based optimizers that merely drive the cost function to a very low value. 
+ Stochastic gradient descent applied to non-convex loss functions is sensitive to the values of the initial parameters.

### Cost functions

+ Our parametric model defines a distribution $p(y | x;\theta)$ and we simply use the principle of maximum likelihood. This means we use the cross-entropy between the training data and the model's predictions as the cost function.
+ The total cost function used to train a neural network will often combine oneof the primary cost functions described here with a regularization term.
+ Most Neural networks are trained using maximum likelihood. This means that the cost function is simply the negative log-likelihood, equivalently described as the cross-entropy between the training data and the model distribution. The cost function is given by
$J(\theta) = -\mathbb{E}_{x,y ~ p_{data}} log p_{model}(y | x)$
+ The expansion of the above equation typically yields some terms that do not depend on the model parameters and may be discarded. If $p_{model}(y | x) = \mathscr{N}(y;f(x;\theta),I)$, then we recover the mean squared error cost, $J(\theta) = \frac{1}{2}\mathbb{E}_{x,y ~ p_{data}}||y - f(x;\theta)||^2 + const$, up to a scaling factor of $\frac{1}{2}$ and a term that doesn't depend on $\theta$. The discarded constant is based on the variance of the Gaussian distribution, which in this case we chose not to parametrize.
+ Specifying a model $p(y | x)$ automatically determines a cost function $log p(y|x)$

### Output units

[TODO]

## Hidden units

[TODO]

## Back-propogation

[TODO]
