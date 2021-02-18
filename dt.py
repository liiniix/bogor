import pandas as pd

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
	
	def read_dataset(self, data_path='kr-vs-kp.data'):
		self.dat = pd.read_csv(data_path, header=None).astype(str)
	
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
			first_entry = D[i].unique()[0]
			partition_1 = D[D[i]==first_entry]
			partition_2 = D[D[i]!=first_entry]
			coeff = partition_1.shape[0]/D.shape[0]
			gini = coeff*self.gini_calc(partition_1, i) + (1-coeff)*self.gini_calc(partition_2, i)
			if gini<smallest_gini:
				smallest_gini = gini
				smallest_gini_index = i
				
		return (smallest_gini_index, D[smallest_gini_index].unique()[0])
	def Decision_Tree_Initiation(self):
		D = self.dat
		self.root = self.Decision_Tree(D)
		return self.root
		
	def Decision_Tree(self, D):

		node = Node()
		
		if len(D.iloc[:,-1].unique()) == 1:
			node.set_leaf(D.iloc[:,-1].unique()[0])
			return node

		if len(D.columns) == 1:
			node.set_leaf(D.iloc[:,-1].value_counts().index[0])
			return node

		index, left_class = self.attribute_selection(D)
		node.index = index
		node.left_class = left_class
		
		partition_left = D[D[index]==left_class]
		partition_right = D[D[index]!=left_class]
		del partition_left[index]
		del partition_right[index]

		if partition_left.shape[0] != 0:
			node.left = self.Decision_Tree(partition_left)
			print(partition_left)
		if partition_right.shape[0] != 0:
			node.right = self.Decision_Tree(partition_right)
			print(partition_right)
		
		return node

classifier = DT_Classifier()

classifier.read_dataset()

n = classifier.Decision_Tree_Initiation()

n.pprint()





