from numpy import exp, array, random, dot
import pandas as pd


class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # ham sigmoid
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # Đạo hàm của hàm Sigmoid.
    # Đây là độ dốc của đường cong Sigmoid.
    # no cho biet do tin cay cua cac weight hien co
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # huan luyen mạng lưới thần kinh thông qua một quá trình thử và sai (trial and error)
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    # The neural network prints its weights
    def print_weights(self):
        print("    Layer 1 (15 neurons, each with 7 inputs):")
        print(self.layer1.synaptic_weights)
        print("    Layer 2 (1 neuron, with 15 inputs):")
        print(self.layer2.synaptic_weights)
        

if __name__ == "__main__":

    #Seed the random number generator
    random.seed(1)

    # Create layer 1 (15 neurons, each with 7 features inputs from a vector)
    layer1 = NeuronLayer(10, 7)

    # Create layer 2 (a single neuron with 15 inputs)
    layer2 = NeuronLayer(1, 10)

    # layer3 = NeuronLayer(1,15)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2)
    # neural_network = NeuralNetwork(layer2, layer3)  


    print("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = pd.read_csv("Data_ANN.csv").to_numpy()
    training_set_outputs = array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]).T

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print("Stage 3) Considering a test situation ?: ")
    testing_array = pd.read_csv("orange_test_data.csv").to_numpy()
    r, c = testing_array.shape
    for i in range(r):
        hidden_state, output = neural_network.think(testing_array[i])
        if output <= 0.5:
            print("strawberry")
        else:
            print("orange")
