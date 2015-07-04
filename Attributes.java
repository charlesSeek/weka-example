
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.experiment.Stats;


public class Attributes {
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource("weather.arff");
		Instances data = source.getDataSet();
		if (data.classIndex()==-1){
			data.setClassIndex(data.numAttributes()-1);
		}
		System.out.println("class index:"+data.classIndex());
		int numAttr = data.numAttributes()-1;
		System.out.println("the number of attributes:"+numAttr);
		for (int i=0;i<numAttr;i++){
			if (data.attribute(i).isNominal()){
				System.out.println("the"+i+" th attribute is numeric");
				int n = data.attribute(i).numValues();
				System.out.println("the "+i+" th attribute has:"+n+" values");
			}
			AttributeStats as = data.attributeStats(i);
			int dc = as.distinctCount;
			System.out.println("the "+i+"th attribute has "+dc+" distinct value");
			if (data.attribute(i).isNumeric()){
				System.out.println("the "+i+"th attribute is numeric");
				Stats s = as.numericStats;
				System.out.println("the "+i+"th attribute has min:"+s.min+" max:"+s.max);
				
			}
		}
		int numInst = data.numInstances();
		for (int j=0;j<numInst;j++){
			Instance instance = data.instance(j);
			double cv = instance.classValue();
			//System.out.println("cv:"+cv);
			System.out.println(instance.classAttribute().value((int) cv));
		}
	}

}
