import java.io.File;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AddClassification;


public class OutputPredictToTestFile {
	public static void main(String[] args) throws Exception{
		/**
		 * load training and testing data
		 */
		DataSource source = new DataSource("iris-train.arff");
		Instances traindata = source.getDataSet();
		traindata.setClassIndex(traindata.numAttributes()-1);
		DataSource source2 = new DataSource("iris-unknown.arff");
		Instances testdata = source2.getDataSet();
		testdata.setClassIndex(testdata.numAttributes()-1);
		/**
		 * training the naive bayes classifier
		 */
		NaiveBayes nb = new NaiveBayes();
		/**
		 * filter the data and output the prediction to test file
		 */
		AddClassification addClass = new AddClassification();
		addClass.setClassifier(nb);
		addClass.setRemoveOldClass(true);
		addClass.setOutputClassification(true);
		addClass.setInputFormat(traindata);
		Filter.useFilter(traindata, addClass);
		Instances newtestdata = Filter.useFilter(testdata, addClass);
		ArffSaver saver = new ArffSaver();
		saver.setInstances(newtestdata);
		saver.setFile(new File("iris-new.arff"));
		saver.writeBatch();
		
	}

}
