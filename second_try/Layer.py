import random, math

input_layer = 0
hidden_layer = 1
output_layer = 2

random_limit = 2.0

k = 50  # número de instâncias por época (?)

delta = 1  # valor que o erro deve diminuir para continuar o backprop

epsilon = 1  # valor que o erro deve chegar para parar o backprop


def sigmoid(x):
    return 1.0/(1 + math.exp(-x))


class Layer:
    def __init__(self, type, neuron_num, inputs_num, alpha):
        self.alpha = alpha
        self.neuron_num = neuron_num
        self.inputs_num = inputs_num
        self.ins_per_neuron = inputs_num/neuron_num
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
        #  mais um peso de bias para cada neurônio, caso não seja input layer
        if self.type is not input_layer:
            self.bias_weights = [random.uniform(-random_limit, random_limit) for i in range(self.neuron_num)]
        else:
            self.bias_weights = []
        self.gradients = [0.0 for i in range(self.inputs_num)]
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
            if self.type is not input_layer:
                sum += self.bias_weights[i]
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

    def gradient(self):
        for i in range(self.inputs_num):
            index = i//self.ins_per_neuron
            self.gradients[i] = self.input[i] * self.deltas[index]

        return

    def update(self):
        for i in range(self.inputs_num):
            self.weights[i] = self.weights[i] - (self.alpha * self.gradients[i])

        #  update bias weights
        for i in range(self.neuron_num):
            self.bias_weights[i] = self.bias_weights[i] - (self.alpha * self.deltas[i])

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
        self.alpha = 0.0

    def initialize(self, dataset, neurons_input_layer, neurons_hidden_layer, neurons_output_layer, layer_num, alpha):
        self.dataset = dataset
        self.neurons_input_layer = neurons_input_layer
        self.neurons_hidden_layer = neurons_hidden_layer
        self.neurons_output_layer = neurons_output_layer
        self.layer_num = layer_num
        self.alpha = alpha

        self.layers.append(Layer(input_layer, self.neurons_input_layer, self.neurons_input_layer, self.alpha))
        #  primeira hidden layer tem um número diferenciado de entradas
        self.layers.append(Layer(hidden_layer, self.neurons_hidden_layer, self.neurons_input_layer, self.alpha))
        for i in range(self.layer_num):
            self.layers.append(Layer(hidden_layer, self.neurons_hidden_layer, self.neurons_hidden_layer * self.neurons_hidden_layer, self.alpha))
        self.layers.append(Layer(output_layer, self.neurons_output_layer, self.neurons_hidden_layer * self.neurons_output_layer, self.alpha))

    def forward_feed(self, train_inst):  # função para propagar as entradas
        for i in range(self.neurons_input_layer):
            self.layers[0].input[i] = train_inst[i]  # passa o exemplo de treinamento para a entrada da inpt layer

        for i in range(self.layer_num):
            self.layers[i].activation()  # calcula a ativação com a entrada (input layer já recebeu a entrada)
            if self.layers[i].type is not output_layer:
                self.layers[i+1].set_input_from_activia(self.layers[i].activia)  # passa a saída da layer atual como entrada da próxima

        return

    def err_func_single(self, y):
        predicted = self.layers[self.layer_num-1].activia
        k = predicted.__len__()
        err = 0.0
        for i in range(k):
            err += - (y[i] * k * math.log10(predicted[i])) - (1-y[i]) * k * (math.log10(1-predicted[i]))

        return err

    def err_func(self):
        n = self.train.__len__()
        err = 0.0
        for inst in self.train:
            err += self.err_func_single(expected)
        err = err/n

        return err

    def calc_deltas(self, expected):
        self.layers[self.layer_num-1].output_delta(expected)
        for i in range(self.layer_num-2, 0, -1):
            self.layers[i].delta(self.layers[i+1].deltas, self.layers[i+1].weights)

        return

    def calc_gradients(self):
        for i in range(self.layer_num-1, 0, -1):
            self.layers[i].gradient()

        return

    def make_updates(self):
        for i in range(self.layer_num - 1, 0, -1):
            self.layers[i].update()

        return

    def train(self):
        i = 0
        err_prev = 0.0
        for inst in self.train:
            self.forward_feed(inst)
            self.calc_deltas(expected)  #  get the expected here somehow TODO
            self.calc_gradients()
            self.make_updates()

            i += 1
            if i == k:
                i = 0
                err = self.err_func()
                if err_prev - err < delta:
                    break
                if err < epsilon:
                    break
                err_prev = err

        return


neural_net = NeuralNet()

if __name__ == '__main__':
    neural_net.initialize('cancer.dat', 3, 5, 1, 2, 0.5)
