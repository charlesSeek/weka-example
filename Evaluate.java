import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class Evaluate {
	public static void main(String[] args) throws Exception{
			DataSource source = new DataSource("iris-train.arff");
			Instances traindata = source.getDataSet();
			traindata.setClassIndex(traindata.numAttributes()-1);
			
			J48 tree = new J48();
			tree.buildClassifier(traindata);
			
			DataSource testsource = new DataSource("iris-test.arff");
			Instances testdata = testsource.getDataSet();
			testdata.setClassIndex(testdata.numAttributes()-1);
			
			Evaluation eval = new Evaluation(traindata);
			eval.evaluateModel(tree, testdata);
			System.out.println(eval.toSummaryString());
			
		}
	

}
