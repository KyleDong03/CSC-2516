# %%
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# # Part 1: Adaptive gradient methods (3 points)
# 
# In the past problem sets, we explored different approaches to implementing gradient descent (hand-written backpropagation, autodiff). In this assignment, we will shift our focus to optimization methods—specifically, algorithms that use gradients to update model weights.
# 
# You will implement the following optimizers:
# 
# * Stochastic Gradient Descent (SGD)
# * SGD with Momentum
# * Adam
# * AdamW

# %% [markdown]
# We are optimizing the following function:
# 
# $$f(x, y) = x^2 + 10y^2$$
# 
# A [contour plot](https://en.wikipedia.org/wiki/Contour_line) of this function is shown below.

# %%
def plot_contour():
    x, y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    plt.contour(x, y, x**2 + 10*y**2, levels=15)
    plt.plot(0, 0, 'rx', ms=20)
plot_contour()


# %% [markdown]
# As you can see, the minimum is at (0, 0), and the curve is much steeper in the y direction than in the x direction.
# 
# Minimize this function using gradient descent. Use the initial point $x = 2, y = 1$. Find one learning rate where optimization diverges and one "good" learning rate that reaches the minimum. For your solution, you should plot the steps taken by your optimizer on top of the contour plot. If you have an array `x` and another `y` which represent the x and y values followed over the course of minimization, you can plot them on top of the contor plot by doing:
# 
# ```Python
# plot_contour()
# plt.plot(x, y, '-')
# ```
# 
# <!-- 1. Show that minimizing this function using [Newton's Method](http://d2l.ai/chapter_optimization/gd.html#newtons-method) will converge to the minimum in a single step.
# 
# 1. Minimize this function using the momentum optimizer. Set the momentum hyperparameter to $0.9$. Can you find a learning rate that allows momentum to converge in less than 20 steps? Now, try optimizing for 100 steps. What is the largest and smallest learning rate you can use when optimizing for 100 steps and still converge near the minumum?
# 
# 1. Implement the Adam optimizer. Use the standard values for all hyperparameters $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-6}, \eta = 0.001$. How many steps does it take for Adam to converge with these hyperparameters? Now, change $\eta$ to the largest value you found that worked for Momentum with 100 steps. Does Adam converge in 100 steps with this value of $\eta$? -->

# %%
def f(x):
    return x[0]**2 + 10*x[1]**2

def get_gradient(x): # df()
    return np.array([2, 20])*x

def minimize(initial, optimizer, N=100):
    x = np.zeros((N, len(initial)))
    x[0, :] = initial
    for n in range(1, N):
        g = get_gradient(x[n-1])
        x[n] = optimizer.step(g, x[n - 1], n)
    return x

# %% [markdown]
# ### A note on optimizer state
# 
# Some optimizers need to remember information from previous optimization steps. Store this information in `self.state`.
# 
# When optimizing multiple model parameters, you may pass `x` and `gradient` as **lists of NumPy arrays**, for example:
# 
# ```python
# x = [W0, b0, W1, b1]
# gradient = [dW0, db0, dW1, db1]
# ```
# 
# In this case, `self.state` should keep corresponding state for each entry in the parameter list. For example, a momentum optimizer could maintain one momentum array for each element of `x`.
# 
# Hint: You do not need to identify parameters using their values or hashes. The order of the parameter list can be used to associate each parameter with its optimizer state.

# %% [markdown]
# ## 1.a. SGD (0.5 point)
# 
# 1. Complete the step function in SGD class.
# 2. Find one "good" learning rate that minimizes the function until the solution is within a tolerance of 1e-3 from (0,0), using 50 steps.
# 3. (not graded) Also, try finding one learning rate where optimization diverges (not graded)!
# 
# You can visualize the convergence by doing
# ```
# your_output = minimize([2, 1], SGD(eta=your_eta), num_steps)
# plot_contour()
# plt.plot(your_output[:, 0], out[:, 1], '-')
# 
# ```

# %%
class SGD:
    def __init__(self, eta):
        self.eta = eta

    def step(self, gradient: np.array, x: np.array, t: int):
        new_x = []
        ############ TODO: Complete the step function ############
        # Return the updated value (array) of x
        new_x = x - self.eta * gradient
        ##########################################################

        return new_x


# %%
############ FOR YOUR INFORMATION ############
plot_contour()

out = minimize([2,1], SGD(0.1), 50)
plt.plot(out[:, 0], out[:, 1], '-')

# %%
##################### FOR ANSWER FOR SGD HYPERPARAMETER #####################
PART1_SGD_HYPERPARAMETER = {
    'eta': np.random.rand(1)[0]
}
#############################################################################

# %% [markdown]
# ## 1.b SGD with momentum (0.7 point)
# 
# 1. Complete the step function in SGDMomentum class.
# 2. Experiment with learning rate and beta parameters that minimizes the function until the solution is within a tolerance of 1e-3 from (0,0), using under 100 steps.
# 3. (not graded) What is the largest and the smallest learning rate you can find when optimizing for 100 steps and still converge near the minumum?

# %%
class SGDMomentum:
    def __init__(self, eta, beta):
        self.eta = eta
        self.beta = beta
        self.state = {}

    def step(self, gradient:np.array, x:np.array, t:int):
        new_x = []
        ############ TODO: Complete the step function ############
        if t == 0:
            self.state["velocity"] = np.zeros_like(x)

        self.state["velocity"] = self.beta * self.state["velocity"] + (1 - self.beta) * gradient

        new_x = x - self.eta * self.state["velocity"]

        ##########################################################
        return new_x

# %%
##################### YOUR ANSWER FOR SGDMOMENTUM HYPERPARAMETER #############
PART1_SGDMOMENTUM_HYPERPARAMETER = {
    'eta': np.random.rand(1)[0],
    'beta': np.random.rand(1)[0],
}
#############################################################################

# %% [markdown]
# ## 1.c Adam (0.8 point)
# 
# 1. Implement the step function in Adam class.
# 2. Try changing the learning rate to the largest value you found that worked for Momentum with 100 steps. Does Adam find the solution in 100 steps with this value? Experiment with learning rate, beta parameters, and epsilon to minimizes the function until the solution is within a tolerance of 1e-3 from (0,0), using 100 steps.

# %%
class Adam:
    def __init__(self, eta, beta1, beta2, epsilon):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.state = {}

    def step(self, gradient:np.array, x:np.array, t:np.array):
        new_x = []
        ################### TODO: Complete the function below  ###################
        if t == 0: 
            self.state["v"] = np.zeros_like(x)
            self.state["s"] = np.zeros_like(x)

        self.state["v"] = self.beta1 * self.state["v"] + (1 - self.beta1) * gradient
        self.state["s"] = self.beta2 * self.state["s"] + (1 - self.beta2) * (gradient ** 2)

        self.state["v_hat"] = self.state["v"] / (1 - self.beta1 ** (t + 1))
        self.state["s_hat"] = self.state["s"] / (1 - self.beta2 ** (t + 1))

        new_x = x - ((self.eta * self.state["v_hat"]) / (self.state["s_hat"] ** 0.5 + self.epsilon))               


        ##########################################################################
        return new_x

# %%
##################### FOR Grading #####################
PART1_ADAM_HYPERPARAMETER = {
    'eta': np.random.rand(1)[0],
    'beta1': np.random.rand(1)[0],
    'beta2': np.random.rand(1)[0],
    'epsilon': np.random.rand(1)[0]
}
#######################################################

# %% [markdown]
# ## 1.d AdamW (1 points)
# 
# While the implementation is similar to Adam, it differs in one key aspect: AdamW applies weight decay in the final parameter update (one line of change!):
# 
# $$
# \theta_{t+1} = \theta_t - \eta \cdot \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
# $$
# 
# Where
# $$
# \theta_t \text{ : current parameters at step } t \\
# \eta \text{ : learning rate} \\
# \hat{m}_t \text{ : bias-corrected first moment estimate (mean of gradients)} \\
# \hat{v}_t \text{ : bias-corrected second moment estimate (uncentered variance of gradients)} \\
# \epsilon \text{ : small constant for numerical stability} \\
# \lambda \text{ : weight decay coefficient (used only in AdamW)}
# $$
# 
# Experiment with learning rate, beta parameters, and epsilon to minimizes the function until the solution is within a tolerance of 1e-3 from (0,0), using 100 steps.
# 
# 1. Complete the step function in AdamW class.
# 2. Experiment with learning rate, beta parameters, and epsilon to reach the desired minimum value.

# %%
class AdamW:
    def __init__(self, eta, beta1, beta2, epsilon, weight_decay):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.state = {}

    def step(self, gradient:np.array, x:np.array, t:np.array):
        new_x = []
        ################### TODO: Complete the function below  ###################
        if t == 0: 
            self.state["v"] = np.zeros_like(x)
            self.state["s"] = np.zeros_like(x)

        self.state["v"] = self.beta1 * self.state["v"] + (1 - self.beta1) * gradient
        self.state["s"] = self.beta2 * self.state["s"] + (1 - self.beta2) * (gradient ** 2)

        self.state["v_hat"] = self.state["v"] / (1 - self.beta1 ** (t + 1))
        self.state["s_hat"] = self.state["s"] / (1 - self.beta2 ** (t + 1))

        new_x = x - self.eta * (((self.state["v_hat"]) / (self.state["s_hat"] ** 0.5 + self.epsilon)) + self.weight_decay * x)               


        ############################################################################
        return new_x

# %%
##################### FOR Grading #####################
PART1_ADAMW_HYPERPARAMETER = {
    'eta': np.random.rand(1)[0],
    'beta1': np.random.rand(1)[0],
    'beta2': np.random.rand(1)[0],
    'epsilon':np.random.rand(1)[0],
    'weight_decay':np.random.rand(1)[0]
}
#######################################################

# %% [markdown]
# # Part 2: Optimization on MLP (Point 1.3)
# 
# In this section, we will reimplement the MLP model from week 2 and train it using the optimizers you have just built. You will observe how different optimizers behave under various weight initializations. We provide the default functions from week 2, as well as the helper functions **step_relu_network_with_optimizer** and **train_relu_mlp_with_optimizer**.
# 
# You should redefine the following functions from week 2:
# * relu
# * relu_derivatives
# * Layer
# * ReLUMLP
# * compute_relu_mlp_forward_passes
# * compute_relu_mlp_parameter_updates
# * compute_relu_mlp_partial_derivatives
# 
# TODOs:
# * Redefine the **update_parameters** function in this notebook to use the optimizer you implemented.
# * Then, experiment with different hyperparameter settings for each optimizer to achieve a final loss < 1 within 100 training steps.
# 

# %%
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

SEED = 42
np.random.seed(SEED)

# %%
### Utils from Week 2
# Don't modify this cell

def softmax(logits: np.array) -> np.array:
    ## Softmax function slightly modified from Homework 1 since data is d_features x N_samples
    logits = logits - np.max(logits, axis=0, keepdims=True)
    exp = np.exp(logits)
    res = exp / np.sum(exp, axis=0, keepdims=True)
    # Remember softmax outputs a vector with same dimensions as the logits
    assert logits.shape == res.shape
    return res


def cross_entropy_loss(probs: np.array, targets: np.array) -> float:
    #  Cross-entropy loss: -1/N * sum y log p (See homework 1)
    n = targets.shape[1]
    loss = -np.sum(targets * np.log(probs + 1e-8)) / n
    return loss

def generate_flower_data(
    n_samples: int = 1000, noise: float = 0.1, num_classes: int = 3, seed: int = 42
):
    rand = np.random.default_rng(seed)
    t = rand.uniform(0, 2 * np.pi, n_samples)

    # Petal shape: radius varies with class count
    r = 1 + 0.3 * np.sin(num_classes * t)

    x = r * np.cos(t) + noise * rand.standard_normal(n_samples)
    y = r * np.sin(t) + noise * rand.standard_normal(n_samples)

    X = np.stack([x, y], axis=1)

    # Assign class based on petal angle region
    labels = ((t % (2 * np.pi)) / (2 * np.pi) * num_classes * 2).astype(
        int
    ) % num_classes
    y = labels.reshape(-1, 1)

    return shuffle(X, y, random_state=seed)


def visualize_classification_data(features, labels, title: str = "Flower Dataset"):
    fig, ax = plt.subplots()
    ax.scatter(x=features[:, 0], y=features[:, 1], c=labels, cmap="viridis")
    plt.title(title)
    return fig, ax


# %%
######################## YOUR (COMPLETED) CODE FROM WEEK 2########################
def relu(x: np.array) -> np.array:
    ##########################################
    ## TODO: Compute ReLU activation,
    res = np.zeros_like(res)
    ##########################################
    assert res.shape == x.shape
    return res

def relu_derivative(x: np.array) -> np.array:
    ##########################################
    ## TODO: Compute the gradient of ReLU with respect to its inputs
    # res = ...
    res = np.zeros_like(res)
    ##########################################
    assert res.shape == x.shape
    return res


class Layer:
    def __init__(self, input_dim, output_dim):
        # Don't change this!
        self.weights = np.random.randn(
            output_dim, input_dim
        ) * np.sqrt(2.0 / input_dim)
        self.biases = np.zeros((output_dim, 1))

    def __call__(self, X):
        ##########################################
        ## TODO: Compute the forward pass of the MLP
        ## Your Code
        # Compute (Wx + b)
        return X
        ###########################################

class ReLUMLP:
    def __init__(self, layer_widths):
        self.layers = []
        ##########################################
        ## TODO: Create the layers of MLP
        ##########################################

    def __call__(self, X):
        res = X
        ##########################################
        ## TODO: Implement forward pass of the ReLU MLP
        ## Hint: Don't forget to apply ReLU after each layer except the last
        ##########################################
        return res

def compute_relu_mlp_forward_passes(
    network: ReLUMLP, inputs: np.array
) -> Tuple[List[np.array], List[np.array]]:
    layer_outputs = [inputs]
    pre_activations = [inputs]
    ##########################################
    ## TODO: Forward pass - compute each layer's pre-activations (h_l) and activations (z_l, after ReLU)
    # and store them for later use
    ##########################################
    assert len(layer_outputs) == len(network.layers) + 1, (
        "Layer outputs should match number of layers"
    )
    assert len(pre_activations) == len(network.layers) + 1, (
        "Pre-activations should match number of layers"
    )
    return layer_outputs, pre_activations

def compute_relu_mlp_partial_derivatives(
    network: ReLUMLP, cost_partials: List[np.array], pre_activations
):
    for ind, (layer, pre_activation) in enumerate(
        zip(reversed(network.layers), reversed(pre_activations))
    ):
        ##########################################
        ## TODO: Compute the error signal propagated to each layer
        ## Hint: Cost partials: d z_l / d z_{l-1} = d z_l / d h_l * d h_l / d z_{l-1}
        pass
    ##########################################
    cost_partials.reverse()
    assert len(cost_partials) == len(network.layers) + 1, (
        "Cost partials should have one more element than layers"
    )

def compute_relu_mlp_parameter_updates(
    cost_partials: List[np.array],
    pre_activations: List[np.array],
    layer_outputs: List[np.array],
    inputs: np.array,
) -> Tuple[List[np.array], List[np.array]]:
    weight_gradients = []
    bias_gradients = []
    ##########################################
    ## TODO: Compute weight gradient
    for cost_partial, layer_output, pre_activation in zip(
        cost_partials[1:], layer_outputs[:-1], pre_activations[:-1]
    ):
        pass
    ##########################################
    ## TODO: Compute bias gradient
    ##########################################
    assert len(weight_gradients) == len(layer_outputs) - 1, (
        "Weight gradients should match number of layers"
    )
    assert len(bias_gradients) == len(layer_outputs) - 1, (
        "Bias gradients should match number of layers"
    )
    return weight_gradients, bias_gradients


##################################################################################

# %%
################ DO NOT MODIFY ################
NUM_CLASSES = 2
N_FEATURES = 2
NUM_SAMPLES = 100
np.random.seed(SEED)

flower_X, flower_y = generate_flower_data(
    num_classes=NUM_CLASSES, n_samples=NUM_SAMPLES, noise=0
)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
flower_X = scaler.fit_transform(flower_X)

visualize_classification_data(flower_X, flower_y)

# %%
############ DO NOT MODIFY ############
np.random.seed(SEED)
inputs = flower_X.T
targets = flower_y
targets = OneHotEncoder().fit_transform(targets).toarray().T

# %%
def update_parameters(
    network: ReLUMLP,
    weight_gradients: List[np.array],
    bias_gradients: List[np.array],
    optimizer,
    timestep
) -> None:

    ## TODO: Perform gradient descent by applying updates to the parameters
    ##########################################
    # Hint: 1. Call the optimizer step function once for the training step,
    #       2. Then assign the returned updated parameters back to the corresponding layers.
    pass
    ##########################################

# %%
def get_optimizer(optimizer_class, **kwargs):
    return optimizer_class(**kwargs)

def step_relu_network_with_optimizer(network, inputs, targets, optimizer, timestep) -> float:
    layer_outputs, pre_activations = compute_relu_mlp_forward_passes(network, inputs)
    probs = softmax(layer_outputs[-1])
    ### Compute loss
    # Number of examples in the dataset
    n = targets.shape[1]
    ##  Cross-entropy: -1/N * sum y log p (See homework 1)
    loss = cross_entropy_loss(probs, targets)

    # backpropagation
    weight_gradients = []
    bias_gradients = []
    cost_partials = [1.0 / n * (probs - targets)]
    compute_relu_mlp_partial_derivatives(network, cost_partials, pre_activations)

    weight_gradients, bias_gradients = compute_relu_mlp_parameter_updates(
        cost_partials, pre_activations, layer_outputs, inputs
    )

    update_parameters(network, weight_gradients, bias_gradients, optimizer, timestep)
    return loss

def train_relu_mlp_with_optimizer(mlp, n_iterations: int, inputs, targets, optimizer):
    losses = []
    for i in range(n_iterations):
        loss = step_relu_network_with_optimizer(mlp, inputs, targets, optimizer, i+1)
        losses.append(loss)
    return losses

# %%
NUM_STEPS=100
NUM_CLASSES=2
################## YOUR HYPERPARAMETER SETTING HERE ########################
BEST_OPTIMIZER_SETTING_CASE1 = {
    'SGD': SGD(eta=np.random.rand(1)[0]),
    'SGDMomentum': SGDMomentum(eta=np.random.rand(1)[0], beta=np.random.rand(1)[0]),
    'Adam': Adam(eta=np.random.rand(1)[0], beta1=np.random.rand(1)[0], beta2=np.random.rand(1)[0], epsilon=np.random.rand(1)[0]),
    'AdamW': AdamW(eta=np.random.rand(1)[0], beta1=np.random.rand(1)[0], beta2=np.random.rand(1)[0], epsilon=np.random.rand(1)[0], weight_decay=np.random.rand(1)[0])
}
############################################################################

f, ax = plt.subplots(figsize=(8,5))
for opt in BEST_OPTIMIZER_SETTING_CASE1:
    np.random.seed(SEED)
    relu_mlp = ReLUMLP([2, 16, NUM_CLASSES])
    myoptimizer = BEST_OPTIMIZER_SETTING_CASE1[opt]
    losses = train_relu_mlp_with_optimizer(mlp=relu_mlp, n_iterations=NUM_STEPS, inputs=inputs, targets=targets, optimizer=myoptimizer)
    print(f"{myoptimizer.__class__.__name__} final loss: {losses[-1]:.4f} Variance of loss: {np.var(losses)}")
    ax.plot(losses, label=myoptimizer.__class__.__name__)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("Loss by optimizer")

# %% [markdown]
# # Part 3: Impact of network initialization on convergence behavior (0.7 point)
# 
# 
# In this section, experiment with different hyperparameter settings to observe how sensitive the model can be under poor network initialization. Then, implement Glorot initialization to see how a more stable weight initialization improves convergence and helps the model train more reliably.
# 
# Essentially, Glorot initialization chooses the range of the weights so that the variance of activations and gradients is roughly the same across layers. This helps prevent common ussies of vanishing gradients or exploding gradients.
# 
# **TODOs:**
# 
# * Implement Glorot initialization:
# $$
# W_{ij} \sim \text{Uniform}\Big[-\text{limit}, \text{limit}\Big], \quad \text{limit} = \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}
# $$
# 
# * Initialize the bias to be 0.
# 

# %%
def init_mlp(seed, init_method, num_classes=NUM_CLASSES):
    """Builds an MLP with all weights = 1 (bad initialization)."""
    np.random.seed(seed)
    relu_mlp = ReLUMLP([2, 16, 16, num_classes])
    relu_mlp = init_method(relu_mlp)
    return relu_mlp

def init_ones(mlp):
    for l in mlp.layers:
        l.weights = np.ones_like(l.weights)
    return mlp

def init_large(mlp):
    for l in mlp.layers:
        l.weights *= 100
    return mlp

def init_glorot(mlp, SEED=42):
    np.random.seed(SEED) ## DO NOT MODIFY THE SEED
    for l in mlp.layers:
    ################## YOUR CODE ##################
        pass
    ###############################################
    return mlp

# %%
NUM_STEPS=100
############################## YOUR HYPERPAREMTER HERE (not graded) ##############################
BEST_OPTIMIZER_SETTING_CASE2={
    'SGD': SGD(eta=np.random.rand(1)[0]),
    'SGDMomentum': SGDMomentum(eta=np.random.rand(1)[0], beta=np.random.rand(1)[0]),
    'Adam': Adam(eta=np.random.rand(1)[0], beta1=np.random.rand(1)[0], beta2=np.random.rand(1)[0], epsilon=np.random.rand(1)[0]),
    'AdamW': AdamW(eta=np.random.rand(1)[0], beta1=np.random.rand(1)[0], beta2=np.random.rand(1)[0], epsilon=np.random.rand(1)[0], weight_decay=np.random.rand(1)[0])
}
######################################################################################

# %%
### CONSTANT INIT
plt.figure(figsize=(8,5))
for opt in BEST_OPTIMIZER_SETTING_CASE2:
    myoptimizer = BEST_OPTIMIZER_SETTING_CASE2[opt]
    model = init_mlp(SEED, init_ones)
    losses = train_relu_mlp_with_optimizer(model, NUM_STEPS, inputs, targets, myoptimizer)
    print(f"{myoptimizer.__class__.__name__} final loss: {losses[-1]:.4f} Variance of loss: {np.var(losses)}")
    plt.plot(losses, label=myoptimizer.__class__.__name__)

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss by optimizer | Constant weight")
plt.show()

# %%
### LARGE INIT
plt.figure(figsize=(8,5))
for opt in BEST_OPTIMIZER_SETTING_CASE2:
    myoptimizer = BEST_OPTIMIZER_SETTING_CASE2[opt]
    model = init_mlp(SEED, init_large)
    losses = train_relu_mlp_with_optimizer(model, NUM_STEPS, inputs, targets, myoptimizer)
    print(f"{myoptimizer.__class__.__name__} final loss: {losses[-1]:.4f} Variance of loss: {np.var(losses)}")
    plt.plot(losses, label=myoptimizer.__class__.__name__)

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss by optimizer | Large weight")
plt.show()

# %%
### Glorot INIT
for opt in BEST_OPTIMIZER_SETTING_CASE2:
    myoptimizer = BEST_OPTIMIZER_SETTING_CASE2[opt]
    model = init_mlp(SEED, init_glorot)
    losses = train_relu_mlp_with_optimizer(model, NUM_STEPS, inputs, targets, myoptimizer)
    print(f"{myoptimizer.__class__.__name__} final loss: {losses[-1]:.4f} Variance of loss: {np.var(losses)}")
    plt.plot(losses, label=myoptimizer.__class__.__name__)

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss by optimizer | Glorot initialization")
plt.show()

# %%



