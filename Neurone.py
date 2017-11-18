import random, math, pickle


class Neurone:
    def __init__(self, ativ, neurones_num, inConnectionList, outConnectionList):
        self.ativ = ativ
        self.delta = 0.0
		
		
def delta(self, deltaListNextLayer):
	for i,delta in enumerate(deltaListNextLayer):
		deltaNeuron += outConnectionList[i].weight*delta*ativ*(1-ativ)
	self.delta = deltaNeuron
	
	return
	
def deltaOutput(self, foreseenClass):

	self.delta = self.ativ - foreseenClass
	
	return
	
def activation(self, ativListPreviousLayer)
	
	self.ativ = 0
	
	for i,activia in enumerate(ativListPreviousLayer)
		self.ativ += inConnectionList[i].weight*activia
	
	return