# Machine Learning Basics (CS Unit 2)

Machine learning is a field of artificial intelligence in which algorithms learn patterns from data instead of following explicitly programmed rules. There are three main paradigms: supervised learning, unsupervised learning and reinforcement learning.

In supervised learning, the model is trained on labelled examples, where each input is paired with the correct output. Classification predicts a discrete category, such as spam or not spam, while regression predicts a continuous value, such as the price of a house. Unsupervised learning works on unlabelled data and discovers structure on its own; clustering with the k-means algorithm is a common example. Reinforcement learning trains an agent to take actions in an environment so as to maximise a cumulative reward.

A model is trained by minimising a loss function, which measures the difference between its predictions and the true values. Gradient descent is the optimisation algorithm that updates the model parameters step by step in the direction that reduces the loss. The size of each step is controlled by the learning rate. If the learning rate is too large the training may diverge, and if it is too small the training becomes very slow.

Overfitting occurs when a model memorises the training data and performs poorly on new data. Underfitting occurs when the model is too simple to capture the pattern. To detect overfitting, the data is split into a training set, a validation set and a test set. Techniques such as regularisation, dropout and early stopping help reduce overfitting.

A neural network is made of layers of connected units called neurons. Each neuron computes a weighted sum of its inputs and passes it through an activation function such as ReLU or sigmoid. Backpropagation is the algorithm that computes the gradient of the loss with respect to every weight so that gradient descent can update them.
