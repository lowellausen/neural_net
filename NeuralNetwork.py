from Layer import Layer


class NeuralNetwork:
    def __init__(self, layer_num, neuronsPerLayer_num, attributes_num, classes_num):
        self.classes_num = classes_num   #  maybe there's a more complicated calculation here
        self.layers = [Layer(attributes_num, 0)]
        for i in range(0, layer_num):
            self.layers.append(Layer(neuronsPerLayer_num, 1))
        self.layers.append(Layer(self.classes_num, 2))
