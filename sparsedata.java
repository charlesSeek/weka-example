/**
 * transfer sparse format arff file to non-sparse format arff file
 * For example:
 * 0,0,0,1,1,0,1
 * {3 1,4,1,6 1}
 */
import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.NonSparseToSparse;


public class sparsedata {
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource("sparse.arff");
		Instances dataset = source.getDataSet();
		NonSparseToSparse sp = new NonSparseToSparse();
		
		sp.setInputFormat(dataset);
		Instances newData = Filter.useFilter(dataset, sp);
		ArffSaver saver = new ArffSaver();
		saver.setInstances(newData);
		saver.setFile(new File("nonsparse.arff"));
		saver.writeBatch();
		
	}

}
