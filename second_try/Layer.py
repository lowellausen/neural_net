import random, math

input_layer = 0
hidden_layer = 1
output_layer = 2

random_limit = 2.0


def sigmoid(x):
    return 1.0/(1 + math.exp(-x))


class Layer:
    def __init__(self, type, neuron_num, inputs_num):
        self.neuron_num = neuron_num
        self.inputs_num = inputs_num
        self.ins_per_neuron = inputs_num/neuron_num
        self.outs_per_neuron =
        self.type = type
        #  inicializa pesos com valor randomico,
        if self.type is input_layer:
            #  input layer tem número de atributos pesos, com valor 1?
            self.weights = [1.0 for i in range(self.inputs_num)]
        if self.type is hidden_layer:
            #  camada oculta tem neurons * neurons pesos
            self.weights = [random.uniform(-random_limit, random_limit) for i in range(self.inputs_num)]
        if self.type is output_layer:
            #  output layer tem número de saídas pesos * neurons na hidden layer anterior, randomicos
            self.weights = [random.uniform(-random_limit, random_limit) for i in range(self.inputs_num)]
        #  mais um peso de bias para cada neurônio, caso não seja output layer
        if self.type is not output_layer:
            self.bias_weights = [random.uniform(-random_limit, random_limit) for i in range(neuron_num)]
        self.gradients = [0.0 for i in range(self.neuron_num)]
        self.deltas = [0.0 for i in range(neuron_num)]
        #  valores de ativação de cada neurônio, saídas
        self.activia = [0.0 for i in range(neuron_num)]
        #  valores de entrada de cada neurônio
        self.input = [0.0 for i in range(self.inputs_num)]

    def set_input_from_activia(self, activia):
        for i in range(self.neuron_num):
            for j in activia:
                index = (i * activia.__len__()) + j
                self.input[index] = activia[j]

        return

    def activation(self):
        for i in range(self.neuron_num):
            sum = 0.0
            for j in range(self.ins_per_neuron):
                index = (i*(self.neuron_num-1))+j  # DANGER cálculo para acessar os índices certo de entrada. certos??
                sum += self.weights[index] * self.input[index]
            self.activia[i] = sigmoid(sum)

        return

    def output_delta(self, expected):
        for i in range(self.neuron_num):
            self.deltas[i] = self.activia[i] - expected[i]

        return

    def delta(self, deltas, weights):
        ln = deltas.__len__()
        for i in range(self.neuron_num):
            delta = 0.0
            for j in range(ln):
                index = (j*self.neuron_num) + i
                delta += deltas[j] * weights[index]

            delta = delta * self.activia[i] * (1 - self.activia[i])
            self.deltas[i] = delta

        return

    def gradient(self, deltas, weights):
        ln = deltas.__len__()
        for i in range(self.neuron_num):
            delta = 0.0
            for j in range(ln):
                index = (j * self.neuron_num) + i
                delta += deltas[j] * weights[index]

            delta = delta * self.activia[i] * (1 - self.activia[i])
            self.deltas[i] = delta

        return

class NeuralNet:
    def __init__(self):
        self.dataset = ""
        self.neurons_input_layer = 0
        self.neurons_hidden_layer = 0
        self.neurons_output_layer = 0
        self.layer_num = 0
        self.layers = []
        self.train = []
        self.test = []

    def initialize(self, dataset, neurons_input_layer, neurons_hidden_layer, neurons_output_layer, layer_num):
        self.dataset = dataset
        self.neurons_input_layer = neurons_input_layer
        self.neurons_hidden_layer = neurons_hidden_layer
        self.neurons_output_layer = neurons_output_layer
        self.layer_num = layer_num

        self.layers.append(Layer(input_layer, self.neurons_input_layer, self.neurons_input_layer))
        #  primeira hidden layer tem um número diferenciado de entradas
        self.layers.append(Layer(hidden_layer, self.neurons_hidden_layer, self.neurons_input_layer))
        for i in range(self.layer_num):
            self.layers.append(Layer(hidden_layer, self.neurons_hidden_layer, self.neurons_hidden_layer * self.neurons_hidden_layer))
        self.layers.append(Layer(output_layer, self.neurons_output_layer, self.neurons_hidden_layer * self.neurons_output_layer))

    def forward_feed(self, train_inst):  # função para propagar as entradas
        for i in range(self.neurons_input_layer):
            self.layers[0].input[i] = train_inst[i]  # passa o exemplo de treinamento para a entrada da inpt layer

        for i in range(self.layer_num):
            self.layers[i].activation()  # calcula a ativação com a entrada (input layer já recebeu a entrada)
            if self.layers[i].type is not output_layer:
                self.layers[i+1].set_input_from_activia(self.layers[i].activia)  # passa a saída da layer atual como entrada da próxima

        return

neural_net = NeuralNet()

if __name__ == '__main__':
    neural_net.initialize('cancer.dat', 3, 5, 1, 2)
