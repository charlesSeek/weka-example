import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.Stacking;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;


public class aggregation {
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource("weather.arff");
		Instances traindata = source.getDataSet();
		traindata.setClassIndex(traindata.numAttributes()-1);
		/**
		 * AdaBoost
		 */
		AdaBoostM1 m1 = new AdaBoostM1();
		m1.setClassifier(new DecisionStump());
		m1.setNumIterations(100);
		m1.buildClassifier(traindata);
		/**
		 * bagging
		 */
		Bagging bagging = new Bagging();
		bagging.setClassifier(new RandomTree());
		bagging.setNumIterations(25);
		bagging.buildClassifier(traindata);
		/**
		 * stacking
		 */
		Stacking stack = new Stacking();
		stack.setMetaClassifier(new Logistic());
		Classifier[] classifiers = {new J48(),new NaiveBayes(),
				new RandomForest()
		};
		stack.setClassifiers(classifiers);
		stack.buildClassifier(traindata);
		/**
		 * voting
		 */
		Vote vote = new Vote();
		vote.setClassifiers(classifiers);
		vote.buildClassifier(traindata);
		
	}
		
	}


