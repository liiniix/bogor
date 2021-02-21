import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

class Node:
	def __init__(self):
		self.left = None
		self.right = None
		self.condition = {}
		self.leaf = False
		self.left_class = ''
		self.index = -1
	def set_leaf(self, class_):
		self.leaf = True
		self.class_ = class_
	
	def pprint(self):
		print(self.leaf)
	

#create a node N ;
#if tuples in D are all of the same class, C, then
#	return N as a leaf node labeled with the class C;
#if attribute list is empty then
#	return N as a leaf node labeled with the majority class in D; // majority voting
#apply Attribute selection method(D, attribute list) to find the "best" splitting criterion;
#label node N with splitting criterion;
#if splitting attribute is discrete-valued and
#	multiway splits allowed then // not restricted to binary trees
#	attribute list = attribute list - splitting attribute; // remove splitting attribute
#for each outcome j of splitting criterion
#	// partition the tuples and grow subtrees for each partition
#	let D j be the set of data tuples in D satisfying outcome j; // a partition
#	if D j is empty then
#		attach a leaf labeled with the majority class in D to node N ;
#	else attach the node returned by Generate decision tree(D j , attribute list) to node N ;
#endfor
#return N ;

class DT_Classifier:
	def __init__(self):
		self.root = Node()
	
	def read_dataset(self, data_path='heart.dat'):
		self.dat = pd.read_csv(data_path, header=None, sep=' ').astype(str).sample(frac=1).reset_index(drop=True)
		self.train = self.dat.head(int(self.dat.shape[0]*.8))
		self.test = self.dat.tail(self.dat.shape[0]-int(self.dat.shape[0]*.2))
	
	def gini_calc(self, D, attr):
		#print(D)
		if D.shape[0]==0:
			return 0

		first_entry = D.iloc[:,-1].unique()[0]

		if len(D.iloc[:,-1].unique()) == 1:
			prob_1 = 1
		else:

			prob_1 = D.iloc[:,-1].value_counts(normalize=True)[first_entry]
		prob_2 = 1 - prob_1
		
		gini = 1. - prob_1**2 - prob_2**2
		return gini
	
	def attribute_selection(self, D):
		smallest_gini = 10000.
		smallest_gini_index = -1.
		for i in D.columns[:-1]:
			D.sort_values(i)
			if len(D[i].unique())<20:
				first_entry = D[i].unique()[0]
				
			else:
				first_entry = D[i].iloc[int(D.shape[0]/2)]
			
			partition_1 = D[D[i]<=first_entry]
			partition_2 = D[D[i]>first_entry]
				
			coeff = partition_1.shape[0]/D.shape[0]
			gini = coeff*self.gini_calc(partition_1, i) + (1-coeff)*self.gini_calc(partition_2, i)
			if gini<smallest_gini:
				smallest_gini = gini
				smallest_gini_index = i
		D.sort_values(smallest_gini_index)
		if len(D[smallest_gini_index].unique())>=20:
			first_entry = D[smallest_gini_index].iloc[int(D.shape[1]/2)]
		else:
			first_entry = D[smallest_gini_index].unique()[0]
			
		return (smallest_gini_index, first_entry)
	def Decision_Tree_Initiation(self):
		D = self.train
		self.root = self.Decision_Tree(D)
		return self.root
		
	def Decision_Tree(self, D):

		node = Node()
		# pruning wrt majority vote
		if D.shape[1]<self.train.shape[1]*.1:
			node.set_leaf(D.iloc[:,-1].value_counts().index[0])
			return node
			
		if len(D.iloc[:,-1].unique()) == 1:
			node.set_leaf(D.iloc[:,-1].unique()[0])
			return node

		if len(D.columns) == 1:
			node.set_leaf(D.iloc[:,-1].value_counts().index[0])
			return node

		index, left_class = self.attribute_selection(D)
		node.index = index
		node.left_class = left_class
		
		partition_left = D[D[index]<=left_class]
		partition_right = D[D[index]>left_class]
		del partition_left[index]
		del partition_right[index]

		if partition_left.shape[0] != 0:
			node.left = self.Decision_Tree(partition_left)
			print(partition_left)
		if partition_right.shape[0] != 0:
			node.right = self.Decision_Tree(partition_right)
			print(partition_right)
		
		return node
	
	def predict(self, data_tuple):
		return self.predict_recursion(data_tuple, self.root)
		
	def predict_recursion(self, data_tuple, node):
		if node is not None:
			if node.leaf == True:
				return node.class_
			if data_tuple[node.index] <= node.left_class:
				return self.predict_recursion(data_tuple, node.left)
			else:
				return self.predict_recursion(data_tuple, node.right)
		else:
			return 'unresolved'
	
	def calc_accuracy(self):
		self.predicted = []
		for index, row in self.test.iloc[:,:-1].iterrows():
			self.predicted.append(self.predict(row))
		self.comparison = pd.DataFrame({'predicted' : self.predicted, 'actual' : self.test.iloc[:,-1]})
		
		self.comparison['res'] = self.comparison.apply(lambda x: 1 if x['predicted']==x['actual'] else 0, axis=1)
		print(self.comparison['res'].value_counts(normalize=True)[1])

	def calc_precision_recall(self):
		print(self.test.iloc[:,-1], self.predicted)
		self.cm = confusion_matrix(self.test.iloc[:,-1], self.predicted, self.test.iloc[:,-1].unique())
		self.recall = np.diag(self.cm) / np.sum(self.cm, axis = 1)
		self.precision = np.diag(self.cm) / np.sum(self.cm, axis = 0)
		print(self.cm, self.test.iloc[:,-1].unique())

classifier = DT_Classifier()

classifier.read_dataset()

n = classifier.Decision_Tree_Initiation()

n.pprint()


def f(n):
	if n is None:
		return
	if n.leaf == False:
		print('l')
		f(n.left)
		print('r')
		f(n.right)
	else:
		pass#print(n.class_)
f(n)

classifier.calc_accuracy()
classifier.calc_precision_recall()
