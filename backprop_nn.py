class NN:
    def __init__(self, n_input, n_hidden, n_output):
        self.n_input = n_input + 1
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.w1 = np.random.normal(scale=0.7, size=(self.n_input*self.n_hidden)).reshape(self.n_input, self.n_hidden)
        self.w2 = np.random.normal(scale=0.7, size=(self.n_hidden*self.n_output)).reshape(self.n_hidden, self.n_output)
        self.output_activation = np.zeros(n_output)
        self.hidden_activation = np.zeros(n_hidden)
        self.input_activation = np.zeros(n_input)
        
    def _sigmoid(self, z):
        return np.tanh(z)
        
    def _dsigmoid(self, z):
        return 1.0 - z**2
    
    def _cost(self, prediction, target):
        return ((prediction - target)**2).sum()
    
    def feed_forward(self):
        """
        Update output vector created by feed-forward propagation of input activations
        """
        self.hidden_activation = self._sigmoid(np.dot(self.input_activation, self.w1))
        self.output_activation = self._sigmoid(np.dot(self.hidden_activation, self.w2))
        
    def back_propagate(self, target, alpha):        
        output_error = target - self.output_activation
        output_delta = output_error * self._dsigmoid(self.output_activation)
        
        hidden_error = np.dot(output_delta, self.w2.T)
        hidden_delta = hidden_error * self._dsigmoid(self.hidden_activation)
        
        self.w2 += alpha * (np.dot(self.hidden_activation.T, output_delta))
        self.w1 += alpha * (np.dot(self.input_activation.T, hidden_delta))
        
    def train(self, data, target, alpha, epochs=50):
        m = data.shape[0]
        
        # add bias to input
        X = np.ones((m, self.n_input))
        X[:, 1:] = data
        
        # turn target into a column vector
        target = target[:, np.newaxis]
        
        for epoch in range(epochs):
            self.input_activation = X
            self.feed_forward()
            self.back_propagate(target, alpha)
            print 'cost: ', self._cost(self.output_activation, target)
            
    def predict(self, data):
        m = data.shape[0]
        self.input_activation = np.ones((m, self.n_input))
        self.input_activation[:, 1:] = data
        self.feed_forward()
        return self.output_activation
        