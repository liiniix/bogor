import math
data = 'kr-vs-kp.data'
from pprint import pprint

import pandas as pd


class NB_Classifier:
	def __init__(self):
		pass
	
	def read_dataset(self, data_path='kr-vs-kp.data'):
		self.dat = pd.read_csv(data_path, header=None).astype(str)
	
	
	def summerize_by_class(self):
		self.classes = self.dat.iloc[:,-1].unique()
		print(self.classes)
		
		self.summary = dict()
		
		for a_class in self.classes:
			separated_dataset = self.dat[self.dat.iloc[:,-1]==a_class]
			if a_class not in self.summary:
				self.summary[a_class] = list()
			
			for column in range(separated_dataset.shape[1]-1):
				self.summary[a_class].append(separated_dataset[column].value_counts(normalize=True).to_dict())
				
	
	def gaussian_liklihood(self, x, mean, std):
		return math.e**(-(x-mean)**2./(2.*std**2.))/(std*math.sqrt(2.*math.pi))
	
	def predict(self, abul=['f','f','f','f','f','f','f','f','t','f','f','f','l','f','n','f','f','t','f','f','f','t','f','t','f','t','f','f','f','f','f','f','f','t','t','n']):
		class_probabilities = dict()
		for a_class in self.classes:
			class_probabilities[a_class] = 1.
			for i, value in enumerate(abul):
				class_probabilities[a_class]*=self.summary[a_class][i][value]
		print(class_probabilities)	


nb = NB_Classifier()

nb.read_dataset()

nb.summerize_by_class()

pprint(nb.summary)
print(nb.predict())
