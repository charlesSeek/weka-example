import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class CrossValidate {
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource("iris.arff");
		Instances dataset = source.getDataSet();
		dataset.setClassIndex(dataset.numAttributes()-1);
		
		NaiveBayes nb = new NaiveBayes();
		int fold = 10;
		int seed = 1;
		Random rand = new Random(seed);
		Instances randData = new Instances(dataset);
		randData.randomize(rand);
		if (randData.classAttribute().isNominal())
			randData.stratify(fold);
		double averagecorrect = 0;
		for (int n=0;n<fold;n++){
			Evaluation eval = new Evaluation(randData);
			Instances train = randData.trainCV(fold, n);
			Instances test = randData.testCV(fold, n);
			nb.buildClassifier(train);
			eval.evaluateModel(nb, test);
			double correct = eval.pctCorrect();
			averagecorrect = averagecorrect + correct;
			System.out.println("the "+n+"th cross validation:"+eval.toSummaryString());
			
		}
		System.out.println("the average correction rate of "+fold+" cross validation: "+averagecorrect/fold);
	}
}
