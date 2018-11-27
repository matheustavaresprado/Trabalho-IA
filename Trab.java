/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package trabalhoia;

import weka.core.Instance;
//import required classes
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stemmers.LovinsStemmer;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Trab{

    public static void main(String args[]) throws Exception{
    //load dataset
    DataSource source = new DataSource("src/trabalhoia/baseia.arff");
    Instances dataset = source.getDataSet();
    //set class index to the last attribute
    dataset.setClassIndex(dataset.numAttributes()-1);

    //the base classifier
    J48 tree = new J48();

    //the filter
    StringToWordVector filter = new StringToWordVector();
    filter.setInputFormat(dataset);
    filter.setIDFTransform(true);
    //filter.setUseStoplist(true);
    LovinsStemmer stemmer = new LovinsStemmer();
    filter.setStemmer(stemmer);
    filter.setLowerCaseTokens(true);

    //Create the FilteredClassifier object
    FilteredClassifier fc = new FilteredClassifier();
    //specify filter
    fc.setFilter(filter);
    //specify base classifier
    fc.setClassifier(tree);
    //Build the meta-classifier
    fc.buildClassifier(dataset);

//    System.out.println(tree.graph());
//    System.out.println(tree);

    Instance dense = new DenseInstance(80);
    dense.setDataset(dataset);
    dense.setValue(0, "'good'");
    //dense.setValue(i, "");
    
    double probabilidades[] = fc.distributionForInstance(dense);
    
    for(int i = 0; i < probabilidades.length; i++){
        System.out.println(i + " - " + probabilidades[i]);
    }
   }
}