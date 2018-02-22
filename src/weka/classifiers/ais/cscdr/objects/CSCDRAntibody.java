/*
 * Created on 23/01/2005
 *
 */
package weka.classifiers.ais.cscdr.objects;

import weka.classifiers.ais.cscdr.objects.Antibody;
import weka.core.Instance;

/**
 * Type: CSCDRAntibody<br>
 * <br>
 * Description: 
 * <br>
 * @author Luiz Otavio V. B. Oliveira, 2012
 *
 */
public class CSCDRAntibody extends Antibody{
    protected final int numClasses;
	
    public final long [] classCounts;
	
    public double fitness;
	

    public CSCDRAntibody(double [] aAttributes, int aClassIndex, int aNumClasses){
    	super(aAttributes, aClassIndex);
    	numClasses = aNumClasses;
    	classCounts = new long[numClasses];
    }
    
    public CSCDRAntibody(Instance aInstance){
        super(aInstance);        
    	numClasses = aInstance.classAttribute().numValues();
    	classCounts = new long[numClasses];
    }
    
    public CSCDRAntibody(CSCDRAntibody aParent){
    	super(aParent);
        this.fitness = aParent.fitness;
    	numClasses = aParent.numClasses;
    	classCounts = new long[numClasses];
    }

    
    public void updateClassCount(Instance aInstance){
    	classCounts[(int)aInstance.classValue()]++;
    }
    
    public void clearClassCounts(){
    	for (int i = 0; i < classCounts.length; i++){
            classCounts[i] = 0;
        }		
    }
    
    public boolean hasMisClassified(){
    	for (int i = 0; i < classCounts.length; i++){
            if(i != (int)getClassification() && classCounts[i] > 0){
                return true;
            }
	}
    	
    	return false;
    }
    
    public boolean canSwitchClass(){
        if(classCounts[(int)getClassification()] == 0){
            if(hasMisClassified()){
                return true;
            }
    	}
        
        int classification = (int)getClassification();
    	for (int i = 0; i < classCounts.length; i++){
            if(i != classification && classCounts[i] > classCounts[classification]) {
                return true;
            }
        }
    	
        // have some instances
    	return false;
    }
    
    public void switchClasses(){
    	long best = -1;
    	int bestIndex = -1;
    	
    	for (int i = 0; i < classCounts.length; i++){
            if(classCounts[i] > best){
                best = classCounts[i];
		bestIndex = i;
            }
        }
    	
    	// assign new class
    	attributes[classIndex] = bestIndex;
    }
    
    public long getTotalCorrect(){
        return classCounts[(int)getClassification()];
    }

    public long[] getClassCounts() {
        return classCounts;
    }
    
    public void calculateAccuracy(){
        fitness = classCounts[(int)getClassification()];
    }
    
    public void calculateFitness(){
    	double totalCorrect =  classCounts[(int)getClassification()];
    	double totalIncorrect = 0.0;
    	for (int i = 0; i < classCounts.length; i++){
            if(i != (int)getClassification()){
                totalIncorrect += classCounts[i];
            }
	}    
    	
    	if(totalCorrect == 0){
            // have nothing correct
            fitness = 0.0;
    	}
    	else {
            fitness = totalCorrect / (totalIncorrect+1);
    	}    	
    }
    
    public double getFitness(){
    	return fitness;
    }

    @Override
    public String toString() {
        String str = "";
        for(int i = 0; i < attributes.length - 1; i++){
            str += attributes[i] + ",";
        }
//        str += attributes[attributes.length-1];
        str += (int)getClassification();
        return str;
    }
}
