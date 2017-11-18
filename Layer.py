from Neurone import Neurone

#  type = 0   input layer
#  type = 1   hidden layer
#  type = 2   output layer


class Layer:
    def __init__(self, neuron_num, layer_type):
        self.layer_type = layer_type
        self.neurones = [Neurone(1.0, neuron_num)]
        for i in range(0, neuron_num):
            self.neurones.append(Neurone(0, neuron_num))
