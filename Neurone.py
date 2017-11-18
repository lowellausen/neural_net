import random, math, pickle


class Neurone:
    def __init__(self, ativ, neurones_num):
        self.ativ = ativ
        self.delta = 0.0
        self.weights = [random.uniform(0.0, 1.0) for i in range(0,neurones_num)]
		
		
def delta(self, deltaListPreviousLayer):
	for i,delta in enumerate(deltaListPreviousLayer):
		deltaNeuron += self.weights[i]*delta*ativ*(1-ativ)
	self.delta = deltaNeuron
	
	return
	
def deltaOutput(self, foreseenClass):

	self.detal = ativ - foreseenClass
	
	return