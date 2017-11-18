from Layer import Layer


class NeuralNetwork:
    def __init__(self, layer_num, neuronsPerLayer_num, attributes_num):
        self.layers = [Layer(neuronsPerLayer_num, 0)]
        self.layers = [self.layers.append(Layer(neuronsPerLayer_num, 1)) for i in range(0, layer_num)]
        self.layers = [Layer(neuronsPerLayer_num, 2)]
