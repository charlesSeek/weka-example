import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class Classifier {
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource("iris.arff");
		Instances dataset = source.getDataSet();
		dataset.setClassIndex(dataset.numAttributes()-1);
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(dataset);
		Evaluation eval = new Evaluation(dataset);
		eval.evaluateModel(nb, dataset);
		System.out.println(eval.toSummaryString());
		
		SMO svm = new SMO();
		svm.buildClassifier(dataset);
		Evaluation eval2 = new Evaluation(dataset);
		eval2.evaluateModel(svm, dataset);
		System.out.println(eval2.toSummaryString());
		
		LibSVM libsvm = new LibSVM();
		String[] options = new String[8];
		options[0] = "-S"; options[1] = "0";
		options[2] ="-K"; options[3] = "2";
		options[4] = "-G"; options[5] = "1.0";
		options[6] = "-C"; options[7] = "1.0";
	    libsvm.setOptions(options);
		libsvm.buildClassifier(dataset);
		Evaluation eval3 = new Evaluation(dataset);
		eval3.evaluateModel(libsvm, dataset);
		System.out.println(eval3.toSummaryString());
	}
}
