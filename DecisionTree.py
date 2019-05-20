from math import log2
import sys
import matplotlib.pyplot as plt
class Node:

	def __init__(self, data, featInd, usedFeatInd):
		self.pathMapDic = {}
		self.classMapDic = {}
		self.dataList = []
		self.usedFeatInd = [featInd]
		self.usedFeatInd.extend(usedFeatInd)
		self.nextNodes = []
		self.featInd = featInd
		if not len(data) == 0:	
			self.__mapData(data)
			self.__setupList()
			self.__addDatas(data)
			self.__setupMap(data)
			self.__entropy = self.__calcEntropy(len(data))
			self.__splitInfo = self.__calcSplitInfo(len(data))
		
	@classmethod
	def load (cls, file_handle):
		featInd = int(file_handle.readline())
		dicSplit = file_handle.readline().split(" ")
		numNodes = int(file_handle.readline())
		ret = Node([], featInd, [])
		for str in dicSplit:
			entrySplit = str.split("=")
			ret.pathMapDic[entrySplit[0]] = int(entrySplit[1])
		for x in range(numNodes):
			ret.nextNodes.extend([TempNode(ret, x)])
		return ret
		
	def __mapData(self, data):
		for x in data:
			currValue = x[0][self.featInd]
			try:
				temp1 = float(currValue)
				temp2 = int(float(currValue))
				if not temp1 == temp2:
					raise TypeError("Feature " + str(currValue) + " is not discrete")
			except ValueError:
				pass
			if not currValue in self.pathMapDic:
				self.pathMapDic[currValue] = len(self.pathMapDic)
				
				
	def __setPathMapDic(self, dictionary):
		self.pathMapDic = dictionary
				
	def __setupList(self):
		for x in range(len(self.pathMapDic)):
			self.dataList.extend([[]])
			self.nextNodes.extend([TempNode(self, x)])
			
	def __setupMap(self, data):
		for x in data:
			currValue = x[1]
			try:
				temp1 = float(currValue)
				temp2 = int(float(currValue))
				if not temp1 == temp2:
					raise TypeError("Label " + str(currValue) + " is not discrete")
			except ValueError:
				pass
			if not currValue in self.classMapDic:
				self.classMapDic[currValue] = len(self.classMapDic)
	
	def __addDatas(self, data):
		for x in data:
			currTrainingExample = x[0]
			path_index = self.pathMapDic[currTrainingExample[self.featInd]]
			self.dataList[path_index].extend([x])
			
	def getEntropy(self):
		return self.__entropy
	
	def getSplitInfo(self):
		return self.__splitInfo
	
	def __calcEntropyAtPath(self, path_id):
		amount_of_classes = [0] * len(self.classMapDic)
		for set in self.getDataForPath(path_id):
			amount_of_classes[self.classMapDic[set[1]]] = amount_of_classes[self.classMapDic[set[1]]] + 1
		entropy = 0
		for x in amount_of_classes:
			if (len(self.getDataForPath(path_id)) == 0) or x == 0:
				continue
			else:
				entropy -= ( (x / len(self.getDataForPath(path_id))) * log2(x / len(self.getDataForPath(path_id))))
		return entropy
	
	def __calcEntropy(self, dataLen):
		entropy = 0
		for path in range(self.getNumPaths()):
			entropy += (len(self.getDataForPath(path)) / dataLen) * self.__calcEntropyAtPath(path)
		return entropy
		
	def __calcSplitInfo(self, dataLen):
		splitInfo = 0
		for path in range(self.getNumPaths()):
			splitInfo -= (len(self.getDataForPath(path)) / dataLen) * log2(len(self.getDataForPath(path)) / dataLen)
		return splitInfo
	
	def getNextNodes(self):
		return self.nextNodes
		
	def setNextNode(self, node, node_index):
		self.nextNodes[node_index] = node
	
	def getDataForPath(self, path_id):
		return self.dataList[path_id]
	
	def getUsedFeatures(self):
		return self.usedFeatInd
	
	def getNumPaths(self):
		return len(self.pathMapDic)
		
	def predict(self, X):
		try:
			valueInd = X[self.featInd]
			nextNode = self.getNextNodes()[self.pathMapDic[valueInd]]
			return nextNode.predict(X)
		except KeyError:
			end_nodes = []
			for node in self.getNextNodes():
				if not type(node) == type(self):
					end_nodes.extend([node])
				else:
					node_featInd = node.featInd
					node_path_map = node.pathMapDic
					node_value = X[node_featInd]
					if node_value in node_path_map:
						return node.predict(X)
				if len(end_nodes) > 0:
					return end_nodes[0].predict(X)
				else:
					return self.getNextNodes()[0].predict(X)
	def store(self, file_handle):
		file_handle.write("Node\n")
		file_handle.write(str(self.featInd) + "\n")
		dictString = ""
		for entry in self.pathMapDic.items():
			dictString += str(entry[0]) + "=" + str(entry[1]) + " "
		file_handle.write(dictString.rstrip() + "\n")
		file_handle.write(str(len(self.nextNodes)) + "\n")
		return True
		
class TempNode:
	
	def __init__(self, parent_node, path_index):
		self.parent = parent_node
		self.index = path_index
	
	def getParent(self):
		return self.parent
	def getIndex(self):
		return self.index
	
	
class EndNode:
	
	def __init__(self, data):
		if not len(data) == 0:
			self.data = data
			self.classMapDic = {}
			self.__mapData(data)
			self.result_class = self.__calc_class(data)
		else:
			self.result_class = -1

	def __mapData(self, data):
		for x in data:
			currValue = x[1]
			if not currValue in self.classMapDic:
				self.classMapDic[currValue] = len(self.classMapDic)
		
	def __calc_class(self, data):
		values = [0] * len(self.classMapDic)
		for x in data:
			values[self.classMapDic[x[1]]] = values[self.classMapDic[x[1]]] + 1
		highest_class_amount = -1
		highest_class_index = -1
		for x in range(len(values)):
			if values[x] > highest_class_amount:
				highest_class_amount = values[x]
				highest_class_index = x
		for entry in self.classMapDic.items():
			if entry[1] == highest_class_index:
				return entry[0]
		return None
			
	@classmethod
	def load (cls, file_handle):
		class_id = (file_handle.readline().rstrip())
		ret = EndNode([])
		ret.result_class = class_id
		return ret
		
	def getClass(self):
		return self.result_class
		
	def predict(self, X):
		return self.getClass()
		
	def store(self, file_handle):
		file_handle.write("End\n")
		file_handle.write(str(self.getClass()) + "\n")
	
class DecisionTree:
	
	def __init__(self):
		self.labelDic = {}
		self.__root = None
		
	def load(self, fileName):
		self.__init__()
		model_file = None
		try:
			model_file = open(fileName, "r")
		except FileNotFoundError:
			return False
		model_file.readline()
		self.__root = Node.load(model_file)
		nodes_to_load = self.__root.getNextNodes()
		while not len(nodes_to_load) == 0:
			temp_nodes = []
			for node in nodes_to_load:
				nodeType = model_file.readline().rstrip()
				if nodeType == "Node":
					node.getParent().setNextNode(Node.load(model_file), node.getIndex())
				else:
					node.getParent().setNextNode(EndNode.load(model_file), node.getIndex())
				if type(node.getParent().getNextNodes()[node.getIndex()]) == type(self.__root):
					temp_nodes.extend(node.getParent().getNextNodes()[node.getIndex()].getNextNodes())
			nodes_to_load = temp_nodes
		
	def store(self, fileName):
		if self.__root == None:
			return False
		file = None
		try:
			file = open(fileName, "w")
		except FileNotFoundError:
			return False
		storeNodes = []
		self.__root.store(file)
		storeNodes.extend(self.__root.getNextNodes())
		while not len(storeNodes) == 0:
			temp_nodes = []
			for x in storeNodes:
				x.store(file)
				if type(x) == type(self.__root):
					temp_nodes.extend(x.getNextNodes())
			storeNodes = temp_nodes
		file.close()
		return False
		
	
	def train(self, X, Y):
		self.__init__()
		dat = []
		for index in range(len(X)):
			dat.extend([[X[index], Y[index]]])
		numFeat = len(X[0])
		self.__map_labels(Y)
		root = None
		growNodes = []
		dataEntropy = self.__entropy(dat)
		trialNodes = []
		for x in range(numFeat):
			trialNodes.extend([Node(dat, x, [])])
		self.__root = self.__selectBestNode(trialNodes, dataEntropy)
		growNodes = self.__root.getNextNodes()
		while not len(growNodes) == 0:
			temp_nodes = []
			for node in growNodes:
				currData = node.getParent().getDataForPath(node.getIndex())
				possible_indexes = [s not in node.getParent().getUsedFeatures() for s in range(numFeat)]
				currTrialNodes = []
				for index in range(len(possible_indexes)):
					if possible_indexes[index]:
						currTrialNodes.extend([Node(currData, index, node.getParent().getUsedFeatures())])
				parent_entropy = node.getParent().getEntropy()
				node.getParent().setNextNode(self.__selectBestNode(currTrialNodes, parent_entropy), node.getIndex())
				if len(node.getParent().getUsedFeatures()) == numFeat or node.getParent().getNextNodes()[node.getIndex()].getEntropy() <= .2:
					node.getParent().setNextNode(EndNode(node.getParent().getDataForPath(node.getIndex())), node.getIndex())
					curr_used = []
					curr_used.extend([node.getParent().getNextNodes()[node.getIndex()].getClass()])
					curr_used.extend(node.getParent().getUsedFeatures())
					continue
				temp_nodes.extend(node.getParent().getNextNodes()[node.getIndex()].getNextNodes())
			growNodes = temp_nodes
		
	def __selectBestNode(self, trial_nodes, parentEntropy):
		if len(trial_nodes) == 0:
			return None
		maxGainRatio = -1
		maxNodeIndex = -1
		for node_index in range(len(trial_nodes)):
			gainSplit = parentEntropy - trial_nodes[node_index].getEntropy()
			if trial_nodes[node_index].getSplitInfo() == 0:
				continue
			gainRatio = gainSplit / trial_nodes[node_index].getSplitInfo()
			if gainRatio > maxGainRatio:
				maxGainRatio = gainRatio
				maxNodeIndex = node_index
		return trial_nodes[maxNodeIndex]
		
	def predict(self, X):
		return self.__root.predict(X)
		
	def __entropy(self, dataset):
		classAmount = [0] * len(self.labelDic)
		for set in dataset:
			classAmount[self.labelDic[set[1]]] = classAmount[self.labelDic[set[1]]] + 1
		entropy = 0
		for x in classAmount:
			entropy -= ( (x / len(dataset)) * log2(x / len(dataset)))
		return entropy
		
	def __map_labels(self, Y):
		for label in Y:
			if not label in self.labelDic.keys():
				self.labelDic[label] = len(self.labelDic)
		
		
def readTrainingData(filename):
	file = open(filename, "r")
	if file == None:
		print("Cannot open file, exiting")
		exit(1)
	X = []
	Y = []
	for line in file:
		if line[0] == '#':
			continue
		split = line.split("\t")
		split[len(split)-1] = split[len(split)-1].rstrip()
		Y.extend([split[0]])
		X.extend([split[1:len(split)-1]])
	file.close()
	return X,Y
def readTestData(fileName):
	file = open(fileName,"r")
	if file == None:
		print("Cannot open file, exiting")
		exit(1)
	X = []
	for line in file:
		if line[0] == '#':
			continue
		split = line.split("\t")
		split[len(split)-1] = split[len(split)-1].rstrip()
		X.extend([split[1:len(split)-1]])
	file.close()
	return X
	
def splitData(fileName,n):
	file=open(fileName)
	count=0
	X=[]
	Y=[]
	for line in file:
		count+=1
		if line[0]=='#':
			continue
		if (count%n)==0:
			split = line.split("\t")
			split[len(split)-1] = split[len(split)-1].rstrip()
			Y.extend([split[0]])
			X.extend([split[1:len(split)-1]])
	file.close()
	#print(X)
	return X,Y
			
	
command=sys.argv[1]
file=sys.argv[2]
tree = DecisionTree()
if command == '-t':
	X,Y = readTrainingData(file)
	tree.train(X, Y)
	tree.store('storage.txt')
	print("training")
if command == '-p':
	tree.load('storage.txt')
	T=readTestData(file)
	pred=open('predictions.txt','w')
	for entry in T:
		#print(entry)
		pred.write(tree.predict(entry))
		pred.write("\n")
	pred.close()
	print("Predict")
if command == '-e':
	file2=sys.argv[3]
	output=open("output.txt",'w')
	out=open(file,'r')
	test=open(file2,'r')
	TP=0
	TN=0
	FP=0
	FN=0
	for line in test:
		line=line.rstrip()
		testln=out.readline().rstrip()
		testln=testln.strip()
		if testln=='1' and line=='1':
			TP+=1
		if testln=='1' and line=='0':
			FP+=1
		if testln=='0' and line=='0':
			TN+=1
		if testln=='0' and line=='1':
			FN+=1
	Accuracy=(TP+TN)/(TP+TN+FP+FN)
	Precision=(TP)/(TP+FP)
	Recall=(TP)/(TP+FN)
	Specificity=(TN)/(FP+TN)
	F=2*(Precision*Recall)/(Precision+Recall)
	output.write("Accuracy="+str(Accuracy)+"\n")
	output.write("Precision="+str(Precision)+"\n")
	output.write("Recall="+str(Recall)+"\n")
	output.write("Specificity="+str(Specificity)+"\n")
	output.write("F="+str(F)+"\n")
	output.close()
if command =='-d':
	truePosX=[]
	falsePosY=[]
	for n in range(1,11):
		X,Y=splitData(file,n)
		tree.train(X,Y)
		tree.store("storage"+str(n)+".txt")
		file2=sys.argv[3]
		file3=sys.argv[4]
		output=open("output"+str(n)+".txt",'w')
		T=readTestData(file3)
		pred=open("predictions"+str(n)+".txt",'w')
		for entry in T:
			pred.write(tree.predict(entry))
			pred.write("\n")
		pred.close()
		out=open("predictions"+str(n)+".txt",'r')
		test=open(file2,'r')
		TP=0
		TN=0
		FP=0
		FN=0
		for line in test:
			line=line.rstrip()
			testln=out.readline().rstrip()
			testln=testln.strip()
			if testln=='1' and line=='1':
				TP+=1
			if testln=='1' and line=='0':
				FP+=1
			if testln=='0' and line=='0':
				TN+=1
			if testln=='0' and line=='1':
				FN+=1
		Accuracy=(TP+TN)/(TP+TN+FP+FN)
		Precision=(TP)/(TP+FP)
		Recall=(TP)/(TP+FN)
		Specificity=(TN)/(FP+TN)
		F=2*(Precision*Recall)/(Precision+Recall)
		truePosX.append(Recall)
		falsePosY.append(1-Specificity)
		output.write("Accuracy="+str(Accuracy)+"\n")
		output.write("Precision="+str(Precision)+"\n")
		output.write("Recall="+str(Recall)+"\n")
		output.write("Specificity="+str(Specificity)+"\n")
		output.write("F="+str(F)+"\n")
		output.close()
	plt.scatter(truePosX,falsePosY)
	plt.show()
