import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMOreg;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class SaveLoadModel {
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource("house.arff");
		Instances traindata = source.getDataSet();
		traindata.setClassIndex(traindata.numAttributes()-1);
		
		SMOreg smo = new SMOreg();
		smo.buildClassifier(traindata);
		
		weka.core.SerializationHelper.write("smo.model", smo);
		
		SMOreg smo2 = (SMOreg) weka.core.SerializationHelper.read("smo.model");
		Evaluation evol = new Evaluation(traindata);
		evol.evaluateModel(smo2, traindata);
		System.out.println(evol.toSummaryString());
		
	}
}
