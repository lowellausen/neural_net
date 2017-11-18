from Neurone import Neurone


class Layer:
    def __init__(self, neuron_num):
        self.neurones = [Neurone(1, neuron_num)]
        self.neurones = [self.neurones.append(Neurone(0, neuron_num)) for i in range(0, neuron_num)]
        