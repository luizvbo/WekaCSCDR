/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.ais.cscdr.objects;

import java.util.List;
import weka.classifiers.ais.cscdr.CSCDR_v5_paperMDLFitness;
import weka.core.Instances;

/**
 * Type: CloneMachine<br>
 * <br>
 * Description: 
 * <br>
 * @author Luiz Otavio V. B. Oliveira, 2012
 *
 */
public class CloneMachine extends Thread{
    protected List<CSCDRAntibodyPool> memoryPool;
    private CSCDR_v5_paperMDLFitness algorithm;
    private double fitMaxMin;
    private double fitMin;
    protected Instances antigens;
        
//    affinityFunction = new DistanceFunction(aAntigens);

    public CloneMachine(List<CSCDRAntibodyPool> memoryPool, Instances antigens, CSCDR_v5_paperMDLFitness algorithm, double fitMaxMin, double fitMin) {
        this.memoryPool = memoryPool;
        this.algorithm = algorithm;
        this.fitMaxMin = fitMaxMin;
        this.fitMin = fitMin;
        this.antigens = antigens;
    }
        
    @Override
    public void run() {
        try{
            for(CSCDRAntibodyPool abPool : memoryPool){ 
                algorithm.cloningAndMutation(abPool, antigens, fitMaxMin, fitMin);
            }
        }
        catch(IndexOutOfBoundsException e){ 
            e.printStackTrace();
        }
    }
}

