/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.ais.cscdr.objects;

import java.util.LinkedList;
import java.util.List;
import weka.classifiers.ais.cscdr.CSCDR_v5_paperMDLFitness;
import weka.core.Instances;

/**
 * Type: ExhibitorOfAntigens<br>
 * <br>
 * Description: 
 * <br>
 * @author Luiz Otavio V. B. Oliveira, 2012
 *
 */
public class ExhibitorOfAntigens extends Thread{
    protected List<CSCDRAntibodyPool> memoryPool;
    protected Instances antigens;
    private CSCDR_v5_paperMDLFitness algorithm;
        
//    affinityFunction = new DistanceFunction(aAntigens);

    public ExhibitorOfAntigens(List<CSCDRAntibodyPool> memoryPool, Instances antigens, CSCDR_v5_paperMDLFitness algorithm) {
        this.memoryPool = memoryPool;
        this.antigens = antigens;
        this.algorithm = algorithm;
    }
        
    @Override
    public void run() {
        try{
            for(CSCDRAntibodyPool abPool : memoryPool){ 
                algorithm.exposeAbPoolToAntigens(antigens, abPool);
            }
        }
        catch(IndexOutOfBoundsException e){ 
            e.printStackTrace();
        }
    }

    public void setMemoryPool(LinkedList<CSCDRAntibodyPool> memoryPool) {
        this.memoryPool = memoryPool;
    }
}
