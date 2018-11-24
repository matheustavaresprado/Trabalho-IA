/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package trabalhoia;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author matheus
 */
public class TrabalhoIA {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        ConverterUtils.DataSource ds = new ConverterUtils.DataSource("src/trabalhoia/base.arff");
        Instances ins = ds.getDataSet();
        
       // System.out.println(ins.toString());
        
        ins.setClassIndex(1);
        
        RandomForest naive = new RandomForest();
        naive.buildClassifier(ins);
        
        
        
        Instance dense = new DenseInstance(80);
        dense.setDataset(ins);
           //dense.setValue(0, "kangaroo");
        //dense.setValue(0, 1);
        
        for(int i = 2; i<80; i++){
            if(i == 10){
                dense.setValue(i, 1);
            }
            else dense.setValue(i, 0);
        }
        
        double probabilidades[] = naive.distributionForInstance(dense);
        
       
        for(int i = 0; i<probabilidades.length; i++){
            System.out.println(i + " - " + probabilidades[i]);
        }
    }
}
