/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package trabalhoia;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.pmml.consumer.Regression;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.pmml.jaxbbindings.DecisionTree;

/**
 *
 * @author matheus
 */
public class TrabalhoIA {

	public static void classificaBaseDados (String nomeArquivo, Integer algoritmo) throws Exception {
		
		ConverterUtils.DataSource ds = new ConverterUtils.DataSource("src/trabalhoia/" + nomeArquivo);
        Instances ins = ds.getDataSet();
         
        ins.setClassIndex(1);
        
        Classifier classificador;
        
        switch(algoritmo) {
        	case 1:
        		System.out.println("Random Forest Results: ");
        		classificador = new RandomForest();
        		classificador.buildClassifier(ins);
                break;
        	case 2:
        		System.out.println("Naive Bayes Results: ");
        		classificador = new NaiveBayes();
        		classificador.buildClassifier(ins);
                break;
//        	case 3:
//        		System.out.println("IBk: ");
//        		classificador = new IBk();
//        		classificador.buildClassifier(ins);
//                break;
                case 4:
                    System.out.println("Regressão linear: ");
                    classificador = new LinearRegression();
                    classificador.buildClassifier(ins);
                break;
                
                case 5:
                    System.out.println("Random tree: ");
                    classificador = new RandomTree();
                    classificador.buildClassifier(ins);
                break;
                
                case 6:
                    System.out.println("Decision Stump: ");
                    classificador = new DecisionStump();
                    classificador.buildClassifier(ins);
                break;
                
                case 7:
                    System.out.println("Decision table: ");
                    classificador = new DecisionTable();
                    classificador.buildClassifier(ins);
                break;
                
            default:
            	System.out.println("Algortimo Inv�lido");
            	return;
        }
        
        Instance dense = new DenseInstance(80);
        dense.setDataset(ins);
        
        for(int i = 2; i<80; i++){
            if(i == 10){
                dense.setValue(i, 1);
            }
            else dense.setValue(i, 0);
        }
        
        double probabilidades[] = classificador.distributionForInstance(dense);
        
        for(int i = 0; i < probabilidades.length; i++){
            System.out.println(i + " - " + probabilidades[i]);
        }
    }
	
    public static void main(String[] args) throws Exception {

    	classificaBaseDados("base_random_forest.arff", 1);
    	classificaBaseDados("base_naive_bayes.arff", 2);
//    	classificaBaseDados("base_naive_bayes.arff", 3);
        classificaBaseDados("base_random_forest.arff", 4);
        classificaBaseDados("base_random_forest.arff", 5);
        classificaBaseDados("base_random_forest.arff", 6);
        classificaBaseDados("base_random_forest.arff", 7);
    	
    }
    
}
