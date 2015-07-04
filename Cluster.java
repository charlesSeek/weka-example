import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class Cluster {
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource("weather.arff");
		Instances traindata = source.getDataSet();
		//traindata.setClassIndex(traindata.numAttributes()-1);
		
		SimpleKMeans kmeans = new SimpleKMeans();
		kmeans.setNumClusters(4);
		kmeans.buildClusterer(traindata);
		
		ClusterEvaluation eval = new ClusterEvaluation();
		eval.setClusterer(kmeans);
		eval.evaluateClusterer(traindata);
		System.out.println(eval.clusterResultsToString());
	}
}
