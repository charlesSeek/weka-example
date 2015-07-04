import java.io.File;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;


public class Normalization {
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource("house.arff");
		Instances dataset = source.getDataSet();
		dataset.setClassIndex(dataset.numAttributes()-1);
		/**
		 * normalize all the attribute values between 0 and 1
		 */
		Normalize normalize = new Normalize();
		normalize.setInputFormat(dataset);
		Instances newdata = Filter.useFilter(dataset, normalize);
		/**
		 * linear regression model
		 */
		LinearRegression lr = new LinearRegression();
		lr.buildClassifier(newdata);
		Evaluation lreval = new Evaluation(newdata);
		lreval.evaluateModel(lr, newdata);
		System.out.println(lreval.toSummaryString());
		/**
		 * store newdata in new arff file
		 */
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(newdata);
		saver.setFile(new File("housenormlize.arff"));
		saver.writeBatch();
		
	}
}
