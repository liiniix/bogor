import math
data = 'iris.data'

import pandas as pd


class NB_Classifier:
	def __init__(self):
		pass
	
	def read_dataset(self, data_path='iris.data'):
		self.dat = pd.read_csv(data, header=None)
	
	
	def summerize_by_class(self):
		self.classes = self.dat.iloc[:,-1].unique()
		print(self.classes)
		
		self.summary = dict()
		
		for a_class in self.classes:
			separated_dataset = self.dat[self.dat.iloc[:,-1]==a_class]
			if a_class not in self.summary:
				self.summary[a_class] = list()
			
			for column in range(separated_dataset.shape[1]-1):
				
				self.summary[a_class].append([separated_dataset[column].mean(), separated_dataset[column].std()])
	
	def gaussian_liklihood(self, x, mean, std):
		return math.e**(-(x-mean)**2./(2.*std**2.))/(std*math.sqrt(2.*math.pi))
	
	def predict(self, abul=[5.0,3.2,1.2,0.2]):
		class_probabilities = dict()
		for a_class in self.classes:
			class_probabilities[a_class] = 1.
			for column, value in enumerate(abul):
				class_probabilities[a_class]*=self.gaussian_liklihood(value, *self.summary[a_class][column])
		print(class_probabilities)	


nb = NB_Classifier()

nb.read_dataset()

nb.summerize_by_class()

print(nb.predict())
