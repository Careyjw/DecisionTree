TO RUN
	python DecisionTree.py "command" "File" "Secondary File" "Third File"
	where:
		"command": to train "-t", to predict "-p", to evaluate "-e", to draw plot of 10 differant ammounts of training data "-d"
		"File": for training "TrainingData8.txt", for predicting "TestDataNoLabel8.txt", for evaluate "predictions.txt", plotting "TrainingData8.txt"
		"Secondary": evaluate "TestDataLabel8.txt", plotting "TestDataLabel8.txt"
		"Third File": plotting "TestDataNoLabel8.txt"
	example code:
		to traing the program: python DecisionTree.py -t TrainingData8.txt
		to predict: python DecisionTree.py -p TestDataNoLabel8.txt
		to evaluate: python DecisionTree.py -e predictions.txt TestDataLabel8.txt
		to plot: python DecisionTree.py -d TrainingData8.txt TestDataLabel8.txt TestDataNoLabel8.txt

	outputs:
		Training outputs storage.txt
		predicting outputs predictions.txt
		evaluating outputs output.txt
		plotting will output storage1.txt predictions1.txt and output1.txt, with 1 - 10 in the file names. As well as plot the ROC Graph
		
		
If you have any questions please let me know.
Thank You,
Jim Carey