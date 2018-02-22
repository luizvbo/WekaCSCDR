/*
 * Created on 23/01/2005
 *
 */
package weka.classifiers.ais.cscdr;

import weka.classifiers.ais.cscdr.objects.Antibody;
import weka.core.Instance;

/**
 * Type: CSCDRAntibody_v2<br>
 * <br>
 * Description: 
 * <br>
 * @author Luiz Otavio V. B. Oliveira, 2012
 *
 */
public class CSCDRAntibody_v2 extends Antibody
{
	protected final int numClasses;
	
	protected final long [] classCounts;
	
	protected double fitness;
	

    public CSCDRAntibody_v2(double [] aAttributes, int aClassIndex, int aNumClasses){
    	super(aAttributes, aClassIndex);
    	numClasses = aNumClasses;
    	classCounts = new long[numClasses];
    }
    
    public CSCDRAntibody_v2(Instance aInstance){
        super(aInstance);        
    	numClasses = aInstance.classAttribute().numValues();
    	classCounts = new long[numClasses];
    }
    
    public CSCDRAntibody_v2(CSCDRAntibody_v2 aParent){
    	super(aParent);
    	numClasses = aParent.numClasses;
    	classCounts = new long[numClasses];
    }

    
    public void updateClassCount(Instance aInstance){
    	classCounts[(int)aInstance.classValue()]++;
    }
    
    void updateClassCount(Instance aInstance, double pertinence) {
        classCounts[(int)aInstance.classValue()] += pertinence;
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
            if(i != classification && classCounts[i] > classCounts[classification])
                return true;
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
    
    public double calculateFitness(){
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
            fitness = (totalCorrect+1) / (totalIncorrect+1);
    	}    	
    	
    	return fitness;
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
        str += attributes[attributes.length-1];
        return str;
    }
}
