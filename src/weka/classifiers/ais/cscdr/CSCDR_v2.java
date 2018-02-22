package weka.classifiers.ais.cscdr;

import weka.classifiers.ais.cscdr.objects.CSCDRAntibody;
import weka.classifiers.ais.cscdr.distance.DistanceFunction;
import weka.classifiers.ais.cscdr.distance.AttributeDistance;
import weka.classifiers.ais.cscdr.objects.Antibody;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;

/**
 * Type: CSCDR_v2<br>
 * <br>
 * Description: 
 * <br>
 * @author Luiz Otavio V. B. Oliveira, 2012
 *
 */
public class CSCDR_v2 extends Classifier implements OptionHandler{
    
    public final static NumberFormat format = new DecimalFormat();
    
    // user paramters
    protected int initialPopulationSize; // N
    protected int totalGenerations; // g
    protected long seed; // r
    protected double clonalScaleFactor; // B
    protected int newAbsPerGeneration; // s	
   
    protected String trainingSummary;
    
    protected LinkedList<CSCDRAntibody_v2> memoryPool;
    protected Random rand;
    protected DistanceFunction affinityFunction;

    protected Instances [] partitions;
    protected int partitionIndex;
    
    // statistics
    protected double [] antibodiesPrunedPerGeneration;
    protected double [] populationSizePerGeneration;
    protected double [] antibodiesWithoutErrorPerGeneration;    
    protected double [] antibodyFitnessPerGeneration;
    protected double [] meanAntibodySwitchesPerGeneration;
    protected double [] selectionSetSizePerGeneration;
    protected double [] trainingClassificationAccuracyPerGeneration;
    protected double [] randomInsertionsPerGeneration;
    protected double [] clonesPerGeneration;
    protected int generationsCompleted;
   
    public CSCDR_v2(){
        // set defaults
        initialPopulationSize = 50;
        totalGenerations = 5;
        seed = -1;
        clonalScaleFactor = 1.0;
        newAbsPerGeneration = 3;

        // TODO: should not be true by default
        m_Debug = true;
    }

    
    protected void prepareStatistics(){
        if(m_Debug){
            antibodiesPrunedPerGeneration = new double[totalGenerations];
            populationSizePerGeneration = new double[totalGenerations];
            antibodiesWithoutErrorPerGeneration = new double[totalGenerations];
            antibodyFitnessPerGeneration = new double[totalGenerations];
            meanAntibodySwitchesPerGeneration = new double[totalGenerations];
            selectionSetSizePerGeneration = new double[totalGenerations];
            trainingClassificationAccuracyPerGeneration = new double[totalGenerations];
            randomInsertionsPerGeneration = new double[totalGenerations];
            clonesPerGeneration = new double[totalGenerations];
    	}
    }
    
    @Override
    public double classifyInstance(Instance aInstance){
    	// expose the system to the antigen
    	CSCDRAntibody_v2 bmu = getBestMatchUnit(aInstance);
    	
//    	if(kNN == 1)    	{
    	return bmu.getClassification();
//    	}
    	
//    	int [] counts = new int[aInstance.classAttribute().numValues()];
//    	// accumumate counts of for k instances
//    	for (int i = 0; i < kNN; i++)
//		{
//    		counts[(int)memoryPool.get(i).getClassification()]++;
//		}
//    	// locate largest
//    	int bestCount = -1;
//    	int bestIndex = -1;
//    	for (int i = 0; i < counts.length; i++)
//		{
//			if(counts[i] > bestCount)
//			{
//				bestCount = counts[i];
//				bestIndex = i;
//			}
//		}
//    	
//    	return bestIndex;
    }
    
    
    protected double classificationAccuracy(Instances aInstances){
        int correct = 0;
    	
    	for (int i = 0; i < aInstances.numInstances(); i++){
            Instance current = aInstances.instance(i);
            CSCDRAntibody_v2 bmu = getBestMatchUnit(current);
            if(bmu.getClassification() == current.classValue()){
                correct++;
            }
	}
    	
    	return ((double)correct / (double)aInstances.numInstances()) * 100.0;
    }
    
    protected String getModelSummary(Instances aInstances){
        StringBuilder buffer = new StringBuilder(1024);

        // data reduction percentage
        double dataReduction = 100.0 * (1.0 - ((double)memoryPool.size() / (double)aInstances.numInstances()));

        buffer.append("Data reduction percentage:...").append(format.format(dataReduction)).append("%\n");
        buffer.append("Total training instances:....").append(aInstances.numInstances()).append("\n");
        buffer.append("Total antibodies:............").append(memoryPool.size()).append("\n");
        buffer.append("\n");

        // determine the breakdown of cells
        int numClasses = aInstances.numClasses();
        int [] counts = new int[numClasses];

        for(CSCDRAntibody_v2 c : memoryPool){
            counts[(int)c.getClassification()]++;
        }	    
        buffer.append(" - Classifier Memory Cells - \n");	   
        for(int i=0; i<counts.length; i++){
            int val = counts[i];
            buffer.append(aInstances.classAttribute().value(i)).append(": ").append(val).append("\n");
        }

        return buffer.toString();
    }
        
    protected int getMemoryCellNumber(){
        return memoryPool.size();
    }
    
    protected String getTrainingSummary(Instances aInstances){
    	StringBuilder b = new StringBuilder(1024);
    	
    	if(m_Debug){
            b.append(" - Training Summary - \n");    	
            b.append("Generations completed:.....................").append(generationsCompleted).append("\n");
            b.append("Antibodies without error per generation:...").append(getStatistic(antibodiesWithoutErrorPerGeneration)).append("\n");
            b.append("Population size per generation:............").append(getStatistic(populationSizePerGeneration)).append("\n");
            b.append("Antibody fitness per generation:...........").append(getStatistic(antibodyFitnessPerGeneration)).append("\n");
            b.append("Antibody class switches per generation:....").append(getStatistic(meanAntibodySwitchesPerGeneration)).append("\n");
            b.append("Selection set size per generation:.........").append(getStatistic(selectionSetSizePerGeneration)).append("\n");
            b.append("Training accuracy per generation:..........").append(getStatistic(trainingClassificationAccuracyPerGeneration)).append("\n");
            b.append("Inserted antibodies per generation:........").append(getStatistic(randomInsertionsPerGeneration)).append("\n");
            b.append("Cloned antibodies per generation:..........").append(getStatistic(clonesPerGeneration)).append("\n");
    	}
        
    	b.append("\n");
    	b.append(" - Classifier Summary - \n");
    	b.append(getModelSummary(aInstances)).append("\n");
    	
    	return b.toString();
    }
    
    
    protected String getStatistic(double [] data){
    	double mean = mean(data);
    	double stdev = stdev(data, mean);
    	return format.format(mean) + " " + "(" + format.format(stdev) + ")";
    }
    
    protected double mean(double [] results){
        double mean = 0.0;
        double sum = 0.0;
        for (int i = 0; i < results.length; i++){
            sum += results[i];
        }
        mean = (sum / results.length);
        return mean;
    }
    
    protected double stdev(double [] results, double mean){
        // standard deviation - square root of the average squared deviation from the mean
        double stdev = 0.0;
        double sum = 0.0;
        for (int i = 0; i < results.length; i++)
        {
            double diff = mean - results[i];
            sum += diff * diff;
        }
        stdev = Math.sqrt(sum / results.length);
        return stdev;
    }
        
    protected void algorithmPreperation(Instances aAntigens){
    	// prepare seed
        if(seed < 0)
            rand = new Random(System.currentTimeMillis());
        else
            rand = new Random(seed);        
        // distance metric
        affinityFunction = new DistanceFunction(aAntigens);
        // prepare statistics
        prepareStatistics();
        // initialise antibody set
        initialiseAntibodyPool(aAntigens);
    }
    
    protected void initialiseAntibodyPool(Instances aAntigens){
    	// randomise the dataset
    	aAntigens.randomize(rand);
        memoryPool = new LinkedList<CSCDRAntibody_v2>();
        // select random antigens
        for (int i = 0; i < initialPopulationSize; i++){
            CSCDRAntibody_v2 antibody = new CSCDRAntibody_v2(aAntigens.instance(i));
            memoryPool.add(antibody);
        }  
    }    
            
    public void buildClassifier(Instances data) throws Exception{    	
        
        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        data = new Instances(data);
        // Verifica se os parâmetros estão corretos
        testParameters(data);
        data.deleteWithMissingClass();
                        
        // prepare the algorithm
        algorithmPreperation(data);        
        // train the system
        for (int generation = 0; generation < totalGenerations; generation++){
            // randomise the dataset
            data.randomize(rand);
            
            // Rate between size of Ab and N
            double sizeRate = memoryPool.size()/(double)initialPopulationSize;
            exposeToAntigens(data, sizeRate, generation);
            
            // perform pruning
            int totalPruned = performPruning(null);
            
// ========================================================== Apenas para testes ===========================================================
// =========================================================================================================================================
//            FileWriter fw = new FileWriter("/home/luiz/Dados/UFRGS/Dissertacao/Publicacoes/CEC/resultados/analise_mutacao/cscdr_.txt", true);
//            PrintWriter pw = new PrintWriter(fw);
//            double fitSum = 0;
//            for(CSCDRAntibody_v2 ab : memoryPool)
//                fitSum += ab.getFitness();
//            pw.println(generation + "," + fitSum / memoryPool.size());
//            //pw.println(trainingInstances.relationName() + ":" +algorithm.getMemoryCellNumber());
//            fw.close();
// =========================================================================================================================================
// =========================================================================================================================================

            insertRandomAntigens(data);

            performCloningAndMutation(new LinkedList<CSCDRAntibody_v2>(memoryPool), data, generation);
                    
                    
            clearAccumulatedHistory();
            
            // statistics
            if(m_Debug)
            {
	            antibodiesPrunedPerGeneration[generation] = totalPruned;
	            populationSizePerGeneration[generation] = memoryPool.size();
	            trainingClassificationAccuracyPerGeneration[generation] = classificationAccuracy(data);
	            generationsCompleted++;
            }
        }        
        // perform final pruning
        performPruning(data);
        //adjust kNN as needed
//        kNN = Math.min(kNN, memoryPool.size());
    }
    
    protected void clearAccumulatedHistory(){
    	for(CSCDRAntibody_v2 a : memoryPool){
            a.clearClassCounts();
    	}
    }
    
    protected void insertRandomAntigens(Instances aPartition){    	
    	// randomise the partition again
    	aPartition.randomize(rand);
    	// perform insertion
    	for (int i = 0; i < newAbsPerGeneration; i++){
            // clone the antigen as an antibody
            CSCDRAntibody_v2 clone = new CSCDRAntibody_v2(aPartition.instance(i));
            // add to pool
            memoryPool.add(clone);
        }
    }
    
    

    
    protected void performCloningAndMutation(LinkedList<CSCDRAntibody_v2> selectedSet, Instances aPartition, int generation){
// ========================================================== Apenas para testes ===========================================================
// =========================================================================================================================================
//        double sum = 0;
//        for(CSCDRAntibody_v2 a : selectedSet){
//    		sum += a.getFitness();
//    	}
// =========================================================================================================================================
// =========================================================================================================================================
        
        
        Collections.sort(selectedSet, new comparator_AbByFitness());
        int numClones = 0;
        for(int i=0; i < selectedSet.size(); i++)
            numClones += (int)Math.round(clonalScaleFactor*selectedSet.size()/(selectedSet.size() - i));

        for(int i=0; i < selectedSet.size(); i++){
            numClones = (int)Math.round(clonalScaleFactor*selectedSet.size()/(selectedSet.size() - i));
            // Parâmetro ro = 3
            double ratio = Math.exp(-3*(i+1)/(double)(selectedSet.size())); // e^(-3*i/n)
            
            for(int j=0; j < numClones; j++){    
// ========================================================== Apenas para testes ===========================================================
// =========================================================================================================================================
                //ratio = selectedSet.get(i).getFitness() / sum;
// =========================================================================================================================================
// =========================================================================================================================================
                CSCDRAntibody_v2 clone = new CSCDRAntibody_v2(selectedSet.get(i));
                mutateClone(clone, ratio, aPartition);
                // add to pool
    		memoryPool.add(clone);
            }
        }
    }  
        
//    protected int performFirstPruning(){
//    	
//        // Ordena os anticorpos pelo fitness
//        Collections.sort(memoryPool, new comparator_AbByFitness());
//        // Remove os |memoryPool|-n piores anticorpos
//        int startSize = memoryPool.size();
//        if(memoryPool.size() - initialPopulationSize > 0){
//            for(int i = initialPopulationSize; i < startSize; i++){
//                memoryPool.removeLast();
//            }
//            return startSize - memoryPool.size();
//        }
//    	return 0;
//    }
    
    private int performPruning(Instances data) {
//        List<NCSCDRAntibody_v2> antibodiesList = new ArrayList<NCSCDRAntibody_v2>();
//        for(int i=0; i < memoryPool.size(); i++){
//            antibodiesList.add(memoryPool.get(i));
//        }
        
        Collections.sort(memoryPool, new comparator_AbByFitness());
        int totalToRemove = memoryPool.size() - initialPopulationSize;
        for(int i=0; i < totalToRemove; i++) memoryPool.removeFirst();
        
        if(data != null){
//            exposeToAntigens(data, -1);            
            int count = 0;
            for (Iterator<CSCDRAntibody_v2> iter = memoryPool.iterator(); iter.hasNext();){
                CSCDRAntibody_v2 a = iter.next();

                if(a.getFitness() <= 1){
    //			if(a.hasMisClassified())

                    iter.remove();
                    count++;
                }
            }
            
        }
        
        return totalToRemove;
        
    }
    
    
    protected void calculatePopulationFitness(int generation){
        for(CSCDRAntibody_v2 a : memoryPool){
            // check for a class switch
            if(a.canSwitchClass()){
                // perform a class switch
        	a.switchClasses();
        	if(m_Debug && generation!=-1){
                    meanAntibodySwitchesPerGeneration[generation]++;
        	}
            }

            // calculate fitness
            double fitness = a.calculateFitness();
            if(m_Debug && generation!=-1){
                antibodyFitnessPerGeneration[generation] += fitness;
            }
        }
        
        if(m_Debug && generation!=-1){
            antibodyFitnessPerGeneration[generation] /= memoryPool.size();
        }
    }
    
    protected void updateSumOfAffinities(Instance aInstance, double sizeRate){
    	try {
            // calculate affinity for population
            calculateAffinity(memoryPool, aInstance);
            // sort by ascending numeric order - best affinity at zero
            Collections.sort(memoryPool);
            int numberOfAbs = (int)sizeRate;
            for(int i=0; i < numberOfAbs; i++){
                memoryPool.get(i).updateClassCount(aInstance, 1);
            }
            memoryPool.get((int)sizeRate).updateClassCount(aInstance, sizeRate - numberOfAbs);
                       
        }catch(Exception e){
            System.out.println("S = " + initialPopulationSize + "\ng = " + totalGenerations + "\nB = " + clonalScaleFactor + "\nd = "  + newAbsPerGeneration);
        }
    }
    
    protected void calculateAffinity(LinkedList<CSCDRAntibody_v2> antibodies, Instance aInstance){
        double [] data = aInstance.toDoubleArray();
        
        for(CSCDRAntibody_v2 a : antibodies){
            double affinity = affinityFunction.calculateDistance(a.getAttributes(), data);
            a.setAffinity(affinity);
        }
    }
    
    
    
    protected void mutateClone(Antibody aClone, double aMutationRate, Instances aPartition){
        double [][] minmax = affinityFunction.getMinMax();
        AttributeDistance[] attribs = affinityFunction.getDistanceMeasures();
        
        double [] data = aClone.getAttributes();
       
        for (int i = 0; i < data.length; i++){
            if(attribs[i].isClass()){
                continue;
            }
            else if(attribs[i].isNominal()){
                data[i] = rand.nextInt(aPartition.attribute(i).numValues());
            }
            else if(attribs[i].isNumeric()){                
                // determine the mutation rate based range
                double range = (minmax[i][1] - minmax[i][0]);
//                range = (range * aMutationRate);
                range = (range * aMutationRate);
                
                // determine bounds for new value based on range
                double min = Math.max(data[i]-(range/2.0), minmax[i][0]);
                double max = Math.min(data[i]+(range/2.0), minmax[i][1]);
                
                // generate new value in VALID range and store
                data[i] = min + (rand.nextDouble() * (max-min));
            }
            else{
                throw new RuntimeException("Unsuppored attribute type!");
            }
        }
    }

    private void exposeToAntigens(Instances data, double sizeRate, int generation) {
        // expose the system to the partition
        for (int j = 0; j < data.numInstances(); j++){            
            // get a data instance
            Instance current = data.instance(j);
            // locate the best match
            updateSumOfAffinities(current, sizeRate);
        }            
        // calculate fitness for the population
        calculatePopulationFitness(generation);   
    }

    private void testParameters(Instances data) {
        if(initialPopulationSize <= 0 || initialPopulationSize > data.numInstances()){
            throw new RuntimeException("The initial population size must be greater than zero and less or equal to the number of training instances.");
    	}
        if(totalGenerations <= 0){
            throw new RuntimeException("The number of generations must be greater than zero.");
        }
        if(clonalScaleFactor <= 0){
            throw new RuntimeException("The clonal scale factor must be greater than zero.");
        }
        if(newAbsPerGeneration < 0){
            throw new RuntimeException("The number of antibodies added by generation must be greater or equal to zero.");
        }
    }

    private CSCDRAntibody_v2 getBestMatchUnit(Instance aInstance) {
        CSCDRAntibody bmu = null;
        // calculate affinity for population
        calculateAffinity(memoryPool, aInstance);
        // sort by ascending numeric order - best affinity at zero
        Collections.sort(memoryPool);
        // retrieve bmu
        return memoryPool.getFirst();
    }

    
    /**
     * Interface para ordenação dos anticorpos por fitness
     */
    public class comparator_AbByFitness implements Comparator<CSCDRAntibody_v2> {
        public int compare(CSCDRAntibody_v2 o1, CSCDRAntibody_v2 o2) {
            if(o1.fitness < o2.fitness){
                return -1;
            }
            else if(o1.fitness > o2.fitness){
                return +1;
            }
            else
                return 0;
        }
    }
    
    
    private final static String [] PARAMETERS = {"N","G","R","B","S"};
    
    private final static String [] DESCRIPTIONS ={"Initial population size (N).",
                                                  "Total generations (G).",
                                                  "Random number generator seed (r).",
                                                  "Clonal scale factor (Beta).",
                                                  "Number of new antibodies added per generation (s)."};
        
    @Override
    public String toString(){
        StringBuffer buffer = new StringBuffer(1000);
        buffer.append("Clonal Selection Classification with  Data Reduction (CSCDR) v1.0.\n");
        
        if(trainingSummary != null){
        	buffer.append("\n");
        	buffer.append(trainingSummary);
        }
        
        return buffer.toString();
    }

    public String globalInfo(){
        StringBuffer buffer = new StringBuffer(1000);
        buffer.append(toString());
        buffer.append("\n\n");

        buffer.append("Jason Brownlee.  " +
        		"[Technical Report].  " +
        		"Clonal Selection Theory & CLONAG - The Clonal Selection Classification Algorithm (CSCA).  " +
        		"Victoria, Australia: Centre for Intelligent Systems and Complex Processes (CISCP), " +
        		"Faculty of Information and Communication Technologies (ICT), " +
        		"Swinburne University of Technology; " +
        		"2005 Jan; " +
        		"Technical Report ID: 2-01.\n");
        buffer.append("\\n");
        buffer.append("http://www.it.swin.edu.au/centres/ciscp/ais/\n");       
        
        
        return buffer.toString();
    }

    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();   // returns the object from weka.classifiers.Classifier

        // attributes
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.DATE_CLASS);

        return result;
    }
    
    @Override
    public Enumeration listOptions(){
        Vector<Option> list = new Vector<Option>(15);

        // add parents options
        Enumeration e = super.listOptions();
        while (e.hasMoreElements()){
            list.add((Option) e.nextElement());
        }

        // add new options
        for (int i = 0; i < PARAMETERS.length; i++){
            Option o = new Option(DESCRIPTIONS[i], PARAMETERS[i], 1, "-" + PARAMETERS[i]);
            list.add(o);
        }

        return list.elements();
    }


    protected double getDouble(String param, String[] options) throws Exception{
        String value = Utils.getOption(param.charAt(0), options);
        if (value == null){
            throw new Exception("Parameter not provided: " + param);
        }

        return Double.parseDouble(value);
    }

    protected int getInteger(String param, String[] options) throws Exception{
        String value = Utils.getOption(param.charAt(0), options);
        if (value == null){
            throw new Exception("Parameter not provided: " + param);
        }

        return Integer.parseInt(value);
    }

    protected long getLong(String param, String[] options) throws Exception{
        String value = Utils.getOption(param.charAt(0), options);
        if (value == null){
            throw new Exception("Parameter not provided: " + param);
        }

        return Long.parseLong(value);
    }

    public void setOptions(String[] options) throws Exception{
        // Initial population size
        String aux = Utils.getOption(PARAMETERS[0], options);
        if (aux.length() != 0) {
            setInitialPopulationSize(Integer.parseInt(aux));
        }

        // Total generations
        aux = Utils.getOption(PARAMETERS[1], options);
        if (aux.length() != 0) {
            setTotalGenerations(Integer.parseInt(aux));
        }

        // Random number generator seed
        aux = Utils.getOption(PARAMETERS[2], options);
        if (aux.length() != 0) {
            setSeed(Long.parseLong(aux));
        }

        // Clonal scale factor
        aux = Utils.getOption(PARAMETERS[3], options);
        if (aux.length() != 0) {
            setClonalScaleFactor(Double.parseDouble(aux));
        }
        
        // Number of new cells added per generation
        aux = Utils.getOption(PARAMETERS[4], options);
        if (aux.length() != 0) {
            setNewAbsPerGeneration(Integer.parseInt(aux));
        }

    }
    
    
    public String[] getOptions(){
        LinkedList<String> list = new LinkedList<String>();

        String[] options = super.getOptions();
        for (int i = 0; i < options.length; i++){
            list.add(options[i]);
        }
        
        list.add("-" + PARAMETERS[0]);
        list.add(Integer.toString(initialPopulationSize));    
        list.add("-" + PARAMETERS[1]);
        list.add(Integer.toString(totalGenerations));
        list.add("-" + PARAMETERS[2]);
        list.add(Long.toString(seed));
        list.add("-" + PARAMETERS[3]);
        list.add(Double.toString(clonalScaleFactor));      
        list.add("-" + PARAMETERS[4]);
        list.add(Integer.toString(newAbsPerGeneration));  

        return list.toArray(new String[list.size()]);
    }
        
    
    public String initialPopulationSizeTipText(){return DESCRIPTIONS[0];}    
    public String totalGenerationsTipText(){return DESCRIPTIONS[1];}
    public String seedTipText(){return DESCRIPTIONS[2];}
    public String clonalScaleFactorTipText(){return DESCRIPTIONS[3];}  
    public String newAbsPerGenerationTipText(){return DESCRIPTIONS[4];}
	
	
    public double getClonalScaleFactor(){
        return clonalScaleFactor;
    }
    
    public void setClonalScaleFactor(double clonalScaleFactor){
        this.clonalScaleFactor = clonalScaleFactor;
    }
    
    public int getInitialPopulationSize(){
        return initialPopulationSize;
    }
    
    public void setInitialPopulationSize(int initialPopulationSize){
        this.initialPopulationSize = initialPopulationSize;
    }
    
    public int getNewAbsPerGeneration(){
        return newAbsPerGeneration;
    }
    
    public void setNewAbsPerGeneration(int newAbsPerGeneration){
        this.newAbsPerGeneration = newAbsPerGeneration;
    }
    
    public long getSeed(){
        return seed;
    }
    
    public void setSeed(long seed){
        this.seed = seed;
    }
    
    public int getTotalGenerations(){
        return totalGenerations;
    }
    
    public void setTotalGenerations(int totalGenerations){
        this.totalGenerations = totalGenerations;
    }
   
    public static void main(String[] argv){
        try{
            System.out.println(Evaluation.evaluateModel(new CSCDR_v2(), argv));
        }
        catch (Exception e){
            System.err.println(e.getMessage());
        }
    }
}
