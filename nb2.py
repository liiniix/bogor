import math
data = 'kr-vs-kp.data'
from pprint import pprint
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


class NB_Classifier:
	def __init__(self):
		pass
	
	def read_dataset(self, data_path='nursery.data'):
		self.dat = pd.read_csv(data_path, header=None).astype(str).sample(frac=1).reset_index(drop=True)
		self.train = self.dat.head(int(self.dat.shape[0]*.7))
		self.test = self.dat.tail(self.dat.shape[0]-int(self.dat.shape[0]*.7))
	
	def summerize_by_class(self):
		self.classes = self.train.iloc[:,-1].unique()
		print(self.classes)
		
		self.summary = dict()
		
		for a_class in self.classes:
			separated_dataset = self.train[self.train.iloc[:,-1]==a_class]
			if a_class not in self.summary:
				self.summary[a_class] = []
			
			for column in range(separated_dataset.shape[1]-1):
				self.summary[a_class].append(separated_dataset[column].value_counts(normalize=True).to_dict())
				
	
	def gaussian_liklihood(self, x, mean, std):
		return math.e**(-(x-mean)**2./(2.*std**2.))/(std*math.sqrt(2.*math.pi))
	
	def predict(self, data_tuple):
		class_probabilities = dict()
		for a_class in self.classes:
			class_probabilities[a_class] = 1.
			for i, value in enumerate(data_tuple):
				if value in self.summary[a_class][i]:
					class_probabilities[a_class]*=self.summary[a_class][i][value]
				else:
					class_probabilities[a_class]*=1e-8
				
		return max(class_probabilities, key=class_probabilities.get)	

	def calc_accuracy(self):
		self.predicted = []
		for index, row in self.test.iloc[:,:-1].iterrows():
			self.predicted.append(self.predict(row))
		self.comparison = pd.DataFrame({'predicted' : self.predicted, 'actual' : self.test.iloc[:,-1]})
		
		self.comparison['res'] = self.comparison.apply(lambda x: 1 if x['predicted']==x['actual'] else 0, axis=1)
		self.comparison['res'].value_counts(normalize=True)[1]

	def calc_precision_recall(self):
		self.cm = confusion_matrix(self.test.iloc[:,-1], self.predicted, self.test.iloc[:,-1].unique())
		self.recall_classwise = np.diag(self.cm) / (np.sum(self.cm, axis = 1) + 1.e-8)  
		self.precision_classwise = np.diag(self.cm) / (np.sum(self.cm, axis = 0) + 1.e-8)
		self.recall = np.mean(self.recall_classwise)
		self.precision = np.mean(self.precision_classwise)
		self.f1 = 2.*self.precision*self.recall/(self.precision+self.recall)
		pprint(self.cm)
		print(self.precision, self.recall, self.f1)

nb = NB_Classifier()

nb.read_dataset()

nb.summerize_by_class()

pprint(nb.summary)
nb.calc_accuracy()
nb.calc_precision_recall()
