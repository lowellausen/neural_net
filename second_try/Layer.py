import random, math

input_layer = 0
hidden_layer = 1
output_layer = 2

random_limit = 1.0
train_size = 0.6
validation_size = 0.2
test_size = 0.2

k = 50  # número de instâncias por época (?)

delta = -1  # valor que o erro deve diminuir para continuar o backprop

epsilon = -1  # valor que o erro deve chegar para parar o backprop


def sigmoid(x):
    return 1.0/(1 + math.exp(-x))


class Layer:
    def __init__(self, type, neuron_num, inputs_num, alpha):
        self.alpha = alpha
        self.neuron_num = neuron_num
        self.inputs_num = inputs_num
        self.ins_per_neuron = inputs_num//neuron_num
        self.type = type
        #  inicializa pesos com valor randomico,
        if self.type is input_layer:
            #  input layer tem número de atributos pesos, com valor 1?
            self.weights = [1.0 for i in range(self.inputs_num)]
        if self.type is hidden_layer:
            #  camada oculta tem neurons * neurons pesos
            self.weights = [random.uniform(0, random_limit) for i in range(self.inputs_num)]
        if self.type is output_layer:
            #  output layer tem número de saídas pesos * neurons na hidden layer anterior, randomicos
            self.weights = [random.uniform(0, random_limit) for i in range(self.inputs_num)]
        #  mais um peso de bias para cada neurônio, caso não seja input layer
        if self.type is not input_layer:
            self.bias_weights = [random.uniform(0, random_limit) for i in range(self.neuron_num)]
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
            for j in range(activia.__len__()):
                index = (i * activia.__len__()) + j
                self.input[index] = activia[j]

        return

    def activation(self):
        for i in range(self.neuron_num):
            sum = 0.0
            for j in range(self.ins_per_neuron):
                index = (i*self.ins_per_neuron)+j  # DANGER cálculo para acessar os índices certo de entrada. certos??
                sum += self.weights[index] * self.input[index]
            if self.type is not input_layer:
                sum += self.bias_weights[i]
                sum = sigmoid(sum)
            self.activia[i] = sum

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
        self.validation = []
        self.alpha = 0.0
        self.err = 0.0
        self.recall = 0.0
        self.acc = 0.0

    def initialize(self, dataset, neurons_input_layer, neurons_hidden_layer, neurons_output_layer, layer_num, alpha):
        self.dataset = dataset
        self.neurons_input_layer = neurons_input_layer
        self.neurons_hidden_layer = neurons_hidden_layer
        self.neurons_output_layer = neurons_output_layer
        self.layer_num = layer_num
        self.alpha = alpha

        self.layers.append(Layer(input_layer, self.neurons_input_layer, self.neurons_input_layer, self.alpha))
        #  primeira hidden layer tem um número diferenciado de entradas
        self.layers.append(Layer(hidden_layer, self.neurons_hidden_layer, self.neurons_input_layer * self.neurons_hidden_layer, self.alpha))
        for i in range(self.layer_num-1):
            self.layers.append(Layer(hidden_layer, self.neurons_hidden_layer, self.neurons_hidden_layer * self.neurons_hidden_layer, self.alpha))
        self.layers.append(Layer(output_layer, self.neurons_output_layer, self.neurons_hidden_layer * self.neurons_output_layer, self.alpha))

    def forward_feed(self, train_inst):  # função para propagar as entradas
        for i in range(self.neurons_input_layer):
            self.layers[0].input[i] = train_inst[i]  # passa o exemplo de treinamento para a entrada da inpt layer

        for i in range(self.layer_num+2):
            self.layers[i].activation()  # calcula a ativação com a entrada (input layer já recebeu a entrada)
            if self.layers[i].type is not output_layer:
                self.layers[i+1].set_input_from_activia(self.layers[i].activia)  # passa a saída da layer atual como entrada da próxima

        return

    def err_func_single(self, y):
        predicted = self.layers[self.layer_num+2-1].activia
        k = predicted.__len__()
        err = 0.0
        for i in range(k):
            err += - (y[i] * math.log1p(predicted[i]-1)) - (1-y[i]) * (math.log1p(-predicted[i]))

        return err

    def err_func(self):
        n = self.train.__len__()
        err = 0.0
        for inst in self.train:
            self.forward_feed(inst[0])
            err += self.err_func_single(inst[1])
        err = err/n

        return err

    def calc_deltas(self, expected):
        self.layers[self.layer_num+2-1].output_delta(expected)
        for i in range(self.layer_num, 0, -1):
            self.layers[i].delta(self.layers[i+1].deltas, self.layers[i+1].weights)

        return

    def calc_gradients(self):
        for i in range(self.layer_num+2-1, 0, -1):
            self.layers[i].gradient()

        return

    def make_updates(self):
        for i in range(self.layer_num+2-1, 0, -1):
            self.layers[i].update()

        return

    def train_net(self):
        i = 0
        err_prev = 0.0
        for inst in self.train:
            self.forward_feed(inst[0])
            self.calc_deltas(inst[1])
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

            print(self.err_func())

        return

    def test_single(self, expected):
        predicted = self.layers[self.layer_num+2-1].activia
        real_class = expected.index(max(expected))
        pred_class = predicted.index(max(predicted))

        if real_class is 0 and pred_class is 0:
            return 'vp'
        if real_class is 0 and pred_class is not 0:
            return 'fn'
        if real_class is not 0 and pred_class is not 0:
            return 'vn'
        if real_class is not 0 and pred_class is 0:
            return 'fp'

    def test_net(self, class_num):
        n = self.test.__len__()
        vp = 0  # verdadeiros positivos
        vn = 0  # verdadeiroa negativos
        fp = 0  # falsos positivos
        fn = 0  # falsos negativos
        for inst in self.test:
            self.forward_feed(inst[0])
            res = self.test_single(inst[1])
            if res in 'vp':
                vp += 1.0
            if res in 'vn':
                vn += 1.0
            if res in 'fp':
                fp += 1.0
            if res in 'fn':
                fn += 1.0
        self.recall = vp/(vp+fn)
        self.acc = (vp+vn)/n

        return

    def read_haberman(self):
        haberman_in = open('haberman.data', 'rU').read().splitlines()

        size = haberman_in.__len__()

        """min = 10000
        max = -10000

        for inst in haberman_in:
            inst = inst.split(',')
            if int(inst[2]) > max:
                max = int(inst[2])
            if int(inst[2]) < min:
                min = int(inst[2])"""

        #  ranges:
        #  attribute 0: 30-83
        #  attribute 1: 58-69
        #  attribute 2: 0-52

        for i in range(size):  # normalizations
            haberman_in[i] = haberman_in[i].split(',')
            haberman_in[i][0] = (float(haberman_in[i][0]) - 30)/53
            haberman_in[i][1] = (float(haberman_in[i][1]) - 58)/11
            haberman_in[i][2] = float(haberman_in[i][2])/52

        for i in range(int(train_size*size)):  # reads trainsize instances of the dataset as train data
            index = random.randint(0, haberman_in.__len__()-1)
            inst = haberman_in.pop(index)
            if int(inst[3]) == 1:
                out = [1.0, 0.0]
            else:
                out = [0.0, 1.0]
            self.train.append(([inst[j] for j in range(0, 3)], out))

        for i in range(int(test_size*size)):  # reads testsize instances of the dataset as test data
            index = random.randint(0, haberman_in.__len__()-1)
            inst = haberman_in.pop(index)
            if int(inst[3]) == 1:
                out = [1.0, 0.0]
            else:
                out = [0.0, 1.0]
            self.test.append(([inst[j] for j in range(0, 3)], out))

        for inst in haberman_in:
            if int(inst[3]) == 1:
                out = [1.0, 0.0]
            else:
                out = [0.0, 1.0]
            self.validation.append(([inst[j] for j in range(0, 3)], out))

        haberman_in.clear()

        return

    def read_cmc(self):
        cmc_in = open('cmc.data', 'rU').read().splitlines()

        size = cmc_in.__len__()

        """min = 10000
        max = -10000

        for inst in cmc_in:
            inst = inst.split(',')
            if int(inst[4]) > max:
                max = int(inst[4])
            if int(inst[4]) < min:
                min = int(inst[4])"""

        #  ranges:
        #  attribute 0: 16-49
        #  attribute 1: 1-4
        #  attribute 2: 1-4
        #  attribute 3: 0-16
        #  attribute 4: 0-1
        #  attribute 5: 0-1
        #  attribute 6: 1-4
        #  attribute 7: 1-4
        #  attribute 8: 0-1

        for i in range(size):  # normalizations
            cmc_in[i] = cmc_in[i].split(',')
            cmc_in[i][0] = (float(cmc_in[i][0]) - 16)/(49-16)
            cmc_in[i][1] = (float(cmc_in[i][1]) - 1)/3
            cmc_in[i][2] = (float(cmc_in[i][2]) - 1)/3
            cmc_in[i][3] = (float(cmc_in[i][3]) - 0) / 16
            cmc_in[i][6] = (float(cmc_in[i][6]) - 1) / 3
            cmc_in[i][7] = (float(cmc_in[i][7]) - 1) / 3

        for i in range(int(train_size*size)):  # reads trainsize instances of the dataset as train data
            index = random.randint(0, cmc_in.__len__()-1)
            inst = cmc_in.pop(index)
            if int(inst[9]) == 1:
                out = [1.0, 0.0, 0.0]
            elif int(inst[9] == 2):
                out = [0.0, 1.0, 0.0]
            else:
                out = [0.0, 0.0, 1.0]
            self.train.append(([inst[j] for j in range(0, 9)], out))

        for i in range(int(test_size*size)):  # reads testsize instances of the dataset as test data
            index = random.randint(0, cmc_in.__len__()-1)
            inst = cmc_in.pop(index)
            if int(inst[9]) == 1:
                out = [1.0, 0.0, 0.0]
            elif int(inst[9] == 2):
                out = [0.0, 1.0, 0.0]
            else:
                out = [0.0, 0.0, 1.0]
            self.test.append(([inst[j] for j in range(0, 9)], out))

        for inst in cmc_in:
            if int(inst[9]) == 1:
                out = [1.0, 0.0, 0.0]
            elif int(inst[9] == 2):
                out = [0.0, 1.0, 0.0]
            else:
                out = [0.0, 0.0, 1.0]
            self.validation.append(([inst[j] for j in range(0, 9)], out))

        cmc_in.clear()

        return

    def read_wine(self):
        wine_in = open('wine.data', 'rU').read().splitlines()

        size = wine_in.__len__()

        """for i in range(1,14):
            min = 10000
            max = -10000
            for inst in wine_in:
                inst = inst.split(',')
                if float(inst[i]) > max:
                    max = float(inst[i])
                if float(inst[i]) < min:
                    min = float(inst[i])
            print(str(min) + '-' + str(max))"""

        #  ranges:
        #  attribute 1: 11.03 - 14.83
        #  attribute 2: 0.74 - 5.8
        #  attribute 3: 1.36 - 3.23
        #  attribute 4: 10.6 - 30.0
        #  attribute 5: 70.0 - 162.0
        #  attribute 6: 0.98 - 3.88
        #  attribute 7: 0.34 - 5.08
        #  attribute 8: 0.13 - 0.66
        #  attribute 9: 0.41 - 3.58
        #  attribute 10: 1.28 - 13.0
        #  attribute 11: 0.48 - 1.71
        #  attribute 12: 1.27 - 4.0
        #  attribute 13: 278.0 - 1680.0

        for i in range(size):  # normalizations
            wine_in[i] = wine_in[i].split(',')
            wine_in[i][1] = (float(wine_in[i][1]) - 11.03)/(14.83-11.03)
            wine_in[i][2] = (float(wine_in[i][2]) - 0.74)/(5.8-0.74)
            wine_in[i][3] = (float(wine_in[i][3]) - 1.36) / (3.23-1.36)
            wine_in[i][4] = (float(wine_in[i][4]) - 10.6) / (30.0-10.6)
            wine_in[i][5] = (float(wine_in[i][5]) - 70.0) / (162.0-70.0)
            wine_in[i][6] = (float(wine_in[i][6]) - 0.98) / (3.88 - 0.98)
            wine_in[i][7] = (float(wine_in[i][7]) - 0.34) / (5.08 - 0.34)
            wine_in[i][8] = (float(wine_in[i][8]) - 0.13) / (0.66 - 0.13)
            wine_in[i][9] = (float(wine_in[i][9]) - 0.41) / (3.58 - 0.41)
            wine_in[i][10] = (float(wine_in[i][10]) - 1.28) / (13.0 - 1.28)
            wine_in[i][11] = (float(wine_in[i][11]) - 0.48) / (1.71 - 0.48)
            wine_in[i][12] = (float(wine_in[i][12]) - 1.27) / (4.0 - 1.27)
            wine_in[i][13] = (float(wine_in[i][13]) - 278.0) / (1680.0 - 278.0)

        for i in range(int(train_size*size)):  # reads trainsize instances of the dataset as train data
            index = random.randint(0, wine_in.__len__()-1)
            inst = wine_in.pop(index)
            if int(inst[0]) == 1:
                out = [1.0, 0.0, 0.0]
            elif int(inst[0] == 2):
                out = [0.0, 1.0, 0.0]
            else:
                out = [0.0, 0.0, 1.0]
            self.train.append(([inst[j] for j in range(1, 14)], out))

        for i in range(int(test_size*size)):  # reads testsize instances of the dataset as test data
            index = random.randint(0, wine_in.__len__()-1)
            inst = wine_in.pop(index)
            if int(inst[0]) == 1:
                out = [1.0, 0.0, 0.0]
            elif int(inst[0] == 2):
                out = [0.0, 1.0, 0.0]
            else:
                out = [0.0, 0.0, 1.0]
            self.test.append(([inst[j] for j in range(1, 14)], out))

        for inst in wine_in:
            if int(inst[0]) == 1:
                out = [1.0, 0.0, 0.0]
            elif int(inst[0] == 2):
                out = [0.0, 1.0, 0.0]
            else:
                out = [0.0, 0.0, 1.0]
            self.validation.append(([inst[j] for j in range(1, 14)], out))

        wine_in.clear()

        return

neural_net = NeuralNet()

if __name__ == '__main__':
    #  def initialize(self, dataset, neurons_input_layer, neurons_hidden_layer, neurons_output_layer, layer_num, alpha):
    neural_net.initialize('cancer.dat', 3, 10, 2, 6, 0.01)
    # neural_net.train = [([1.5], [1.0])]
    #  neural_net.layers[1].weights = [0.39]
    #  neural_net.layers[2].weights = [0.94]
    #  neural_net.layers[1].bias_weights = [0.0]
    #  neural_net.layers[2].bias_weights = [0.0]
    neural_net.read_haberman()
    for i in range(1):
        neural_net.train_net()
    neural_net.test_net(2)
    print('Acc = ' + str(neural_net.acc) + '  Recall = ' + str(neural_net.recall))