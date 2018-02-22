/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.ais.cscdr.objects;

import java.io.Serializable;
import java.util.LinkedList;

/**
 * Type: CSCDRAntibodyPool<br>
 * <br>
 * Description: 
 * <br>
 * @author Luiz Otavio V. B. Oliveira, 2012
 *
 */
public class CSCDRAntibodyPool implements Serializable{
    private LinkedList<CSCDRAntibody> abList;
    public double fitness;
    private double accuracyOverTraining;
    private int numberOfClones;
    
    public CSCDRAntibodyPool() {
        abList = new LinkedList<CSCDRAntibody>();
    }

    public CSCDRAntibodyPool(LinkedList<CSCDRAntibody> abList, double fitness) {
        this.abList = abList;
        this.fitness = fitness;
    }
    
    public CSCDRAntibodyPool(CSCDRAntibodyPool abPool){
        this.fitness = abPool.getFitness();
        this.abList = new LinkedList<CSCDRAntibody>();
        if(abPool != null && abPool.size() > 0){
            for(CSCDRAntibody ab : abPool.abList){
                abList.add(new CSCDRAntibody(ab));
            }
        }
    }
    
    public double getFitness() {
        return fitness;
    }

    public LinkedList<CSCDRAntibody> getAbList() {
        return abList;
    }

    public void setFitness(double fitness) {
        this.fitness = fitness;
    }
    
    public int size(){
        if(abList != null) {
            return abList.size();
        }
        else {
            return -1;
        }
    }

    public void clearFitness(){
        fitness = 0;
    }

    public void setAccuracyOverTraining(double accuracyOverTraining) {
        this.accuracyOverTraining = accuracyOverTraining;
    }

    public double getAccuracyOverTraining() {
        return accuracyOverTraining;
    }
    
    @Override
    public String toString() {
        return "Fit=" + fitness;
    }

    public int getNumberOfClones() {
        return numberOfClones;
    }

    public void setNumberOfClones(int numberOfClones) {
        this.numberOfClones = numberOfClones;
    }
}
