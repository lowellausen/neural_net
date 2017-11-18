import random, math, pickle

class Neurone:
    def __init__(self, ativ, neurones_nbr):
        self.ativ = ativ
        self.weights = [random.uniform(0.0, 1.0) for i in range(0,neurones_nbr)]
