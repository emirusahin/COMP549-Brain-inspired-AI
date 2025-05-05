Part 1 – Answering the questions (20 marks)
First, open the code and run it. Once you see the output plot, start working backwards. How
is that plot generated? What is it showing? How are those values calculated, etc.?
Now, answer the following questions:
• Question 1 (8 marks): Can you successfully identify the eight aspects of a PDP model in
the code? Which variables/lines of code represent/calculate which parts? Remember,
the eight aspects we went over in Week 3 are:
1. A set of processing units:
Even though the set of processing units (neurons) are not explicitly defined as objects, they are defined through the W0 and W1 matrices. 
self.N is the number of processing units in the hidden layer, and self.task.noutputs() is the number of processing units in the output layer.

There are 4 neurons in the first layer,
N neurons in the hidden layer ,
3 neurons in the output layer
```
self.N    = N # number of hidden layer units
self.W0   = random.normal(scale=self.alpha,size=(self.N, self.task.ninputs()+1)) # random initialize the input to hidden layer weights, with self.N rows and self.task.ninputs()+1 columns
self.W1   = random.normal(scale=self.alpha,size=(self.task.noutputs(),self.N+1)) # random initialize the hidden to output layer weights with self.task.noutputs() rows and self.N+1 columns
```
2. A state of activation for each unit:
tensor0 and tensor1, in each tensor the row i represents the state of activation of the i-th unit in the hidden and output layer respectively. 
tensor0[0,j] would be the activation of the first neuron in the hidden layer for the jth input.
```def forward(self):
        """
        Tensors are states of activations. 
        """
     
        # myfunction will make it so that each activation will be between 0 and 1
        tensor0 = myfunction(np.dot(self.W0,helper(self.task.inputs))) # state of activation of the hidden layer neurons
    
        tensor1 = myfunction(np.dot(self.W1,helper(tensor0))) # state of activation of the output layer neurons
 ```
3. An output function for each unit:
While some PDP models have explicitly different state of activation and output functions, in this one we don't have it. 
Therefore, output function is basically a identity matrix.
4. A pattern of connectivity between units
```
self.W0   = random.normal(scale=self.alpha,size=(self.N, self.task.ninputs()+1))
self.W1   = random.normal(scale=self.alpha,size=(self.task.noutputs(),self.N+1))
```
5. A propagation rule for sending activity between units:
Our propagation rule is to dot product activation of sending unit i with Wi,j and sum it up for all i. 

6. An activation rule for combining propagated activity with current activity:
Clipping function (myfunction) makes it so that no state of action can be less than 0 or more than 1.
```def myfunction(X):
    """
    If a number in X is less than 0, it is replaced by 0. If a number in X is greater than 1, it is replaced by 1.
    """
    
    Y = X
    mask = (X < 0)
    Y[mask] = 0
    mask = (X > 1)
    Y[mask] = 1
    return Y
 ```
7. A learning rule:
It is defined under one_loop function, which defines a single loop of "training" or "learning". 
Almost gradient descent but it is missing a partial derivative.
```
error    = self.task.outputs - tensor1
delta_W0 = np.dot(np.dot(self.W1[:,:-1].transpose(),error),helper(self.task.inputs).transpose())
delta_W1 = np.dot(error,helper(tensor0).transpose())
self.W0 += self.eta*delta_W0  # update hidden layer weights
self.W1 += self.eta*delta_W1  # update output layer weights
```
To evaluate how well the model is learning over "epochs" or loops, we use mean squared error which is defined under "important_function".

8. An environment:
Set of inputs.

• Question 2 (5 marks): Can you describe in your own words what computation is being calculated and what algorithm is implement to perform this calculation? How did you reach this conclusion?
◦ Hint: Consider what space the units are operating in at each stage of processing.
- Our inputs are given in 4 dimensions and outputs are spitted out in 3 dimensions. We are using matrix multiplication (a.k.a. linear projection) with the bias and blow the input's position up to either side of the decision boundary depending on the control bit.
- The computation is that we use the first bit of the input (control bit) to decide whether to invert the remaining bits. 

• Question 3 (2 marks): Can you predict what the algorithm will do with a novel input?
◦ Hint: Consider what a “soft” version of the computation would look like.
- If we have a novel input with first bit = 0, the remaining 3 bits will be the outputs. If the first bit = 1, then the remaining bits will be inverted. 

• Question 4 (3 marks): Answer these three sub-questions by playing around with the last
cell of the notebook and looking at the first pieces of code:

◦ How does the network’s behaviour change as you change the value of N from very
   low to very high? Why would it change in this way?
- When you have a very low N (N = number of neurons in hidden layer), your model will have a hard time learning patterns. When you have a high N, your model will be to learn more complex patterns. But if you go overbaord and have a very high N, your model will start to memorize the patterns and will overfit.

◦ What happens if you set eta to be very high? Why would this be the impact?
- If eta is too little, the learning will be very slow. If eta is too high, the learning the network likely won't convert to an optimal solution as the step sizes will be too large and the network will keep overshooting the optimal solution (which is the minima of the loss function).

◦ What is the helper function doing? Why do we need it?
- Helper function is used to add a bias term to a given matrix. Bias helps us have decision planes that doesn't have to go through the origin.