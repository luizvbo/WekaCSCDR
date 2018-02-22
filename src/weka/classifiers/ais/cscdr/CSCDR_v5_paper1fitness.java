package weka.classifiers.ais.cscdr;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.ais.cscdr.distance.AttributeDistance;
import weka.classifiers.ais.cscdr.distance.DistanceFunction;
import weka.classifiers.ais.cscdr.objects.Antibody;
import weka.classifiers.ais.cscdr.objects.CSCDRAntibody;
import weka.classifiers.ais.cscdr.objects.CSCDRAntibodyPool;
import weka.core.*;

/**
 * Type: CSCDR_v5_paper1fitness <br>
 * Date: 19/09/2012 <br>
 * <br>
 * 
 * Description: V4 com mudancas: 
 * <br> - Cada incividuo representa um conjunto de prototipos; 
 * <br> - Clonagem baseada no fitness do individuo; 
 * <br> - Mutacao baseada no fitness do gene. 
 * 
 * @author Luiz Otavio Vilas Boas Oliveira
 */
public class CSCDR_v5_paper1fitness extends Classifier implements OptionHandler{
    
    public final static NumberFormat format = new DecimalFormat();
    
    // user paramters
    private int chromosomeSize; // C
    private int populationSize; // P
    private int totalGenerations; // G
    private long seed; // S
    private double clonalScaleFactor; // B
    private double newAbsPerGeneration; // D	
    protected int fitnessMode; // F
   
    private final static Tag [] TAGS_FITNESS_MODE ={
        new Tag(1, "CSCDR based fitness"),
	new Tag(2, "Accuracy based fitness")
    };
    
    private LinkedList<CSCDRAntibody> memoryCells;
    protected String trainingSummary;
    
    protected LinkedList<CSCDRAntibodyPool> memoryPool;
//    protected LinkedList<Integer> antigenPool;
    protected Random rand;
    protected DistanceFunction affinityFunction;

    protected Instances [] partitions;
    protected int partitionIndex;
   
    public CSCDR_v5_paper1fitness(){
        // set defaults
        chromosomeSize = 35;
        populationSize = 40;
        totalGenerations = 50;
        seed = -1;
        clonalScaleFactor = 0.5;
        newAbsPerGeneration = 0.1;
        fitnessMode = 1;

        // TODO: should not be true by default
        m_Debug = false;
    }
    
    @Override
    public double classifyInstance(Instance aInstance){
    	// expose the system to the antigen
    	CSCDRAntibody bmu = selectBestMatchingUnit(aInstance, memoryCells);
    	
    	return bmu.getClassification();
    }
    
    
//    protected double classificationAccuracy(Instances aInstances){
//        int correct = 0;
//    	
//    	for (int i = 0; i < aInstances.numInstances(); i++){
//            Instance current = aInstances.instance(i);
//            CSCDRAntibody bmu = selectBestMatchingUnit(current);
//            if(bmu.getClassification() == current.classValue()){
//                correct++;
//            }
//	}
//    	
//    	return ((double)correct / (double)aInstances.numInstances()) * 100.0;
//    }
    
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

        for(CSCDRAntibody c : memoryPool.get(0).getAbList()){
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
        if(getSeed() < 0)
            rand = new Random(System.currentTimeMillis());
        else
            rand = new Random(getSeed());      
        // distance metric
        affinityFunction = new DistanceFunction(aAntigens);
        // initialise antibody set
        initialiseAntibodyPool(aAntigens);
    }
    
    
    protected CSCDRAntibodyPool generateNewAntibody(Instances aAntigens){
        CSCDRAntibodyPool ab = new CSCDRAntibodyPool();
        LinkedList<Integer> antigenPool = new LinkedList<Integer>();
        
        for(int j=0; j < aAntigens.numInstances(); j++){
            antigenPool.add(j);
        }
        Collections.shuffle(antigenPool);
        
        double gaus = rand.nextGaussian();
        
        // Normal distribution defines the size of the chromosome
        int cSize = (int)Math.round((1-Math.abs(gaus % 3)/3)*chromosomeSize);
        if (cSize == 0) cSize = 1;
        
        for (int i = 0; i < cSize; i++){
            // clone the antigen as an antibody
            CSCDRAntibody clone = new CSCDRAntibody(aAntigens.instance(antigenPool.remove(0)));
            // add to pool
            ab.getAbList().add(clone);
        }
        return ab;
    }
    
    protected void initialiseAntibodyPool(Instances aAntigens){
        memoryPool = new LinkedList<CSCDRAntibodyPool>();
        for (int i=0; i < populationSize; i++){
            memoryPool.add(generateNewAntibody(aAntigens));
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
        // Debug mode
        StringBuilder statistics = new StringBuilder();
        
        // train the system
        for (int generation = 0; generation < getTotalGenerations(); generation++){
            // randomise the dataset
            data.randomize(rand);
              
            exposeToAntigens(data);      
            
            // perform pruning
            performPruning();
            
            if(m_Debug) {
                statistics.append(generation + "," + memoryPool.getLast().getFitness() + "," + memoryPool.getLast().size() +  "\n");
            }
            
            performCloningAndMutation(data);
                    
            insertRandomAntigens(data);   
            
            clearAccumulatedHistory();
        }        
        exposeToAntigens(data);
        // perform final pruning
//        performPruning();
        //adjust kNN as needed
//        kNN = Math.min(kNN, memoryPool.size());
        // Imprime as células de memória na tela
        //printMemoryCells();     
        
        // Defines the memory pool as the best one from abPool
        memoryCells = selectBestAntibodyPool();
        if(m_Debug){
            statistics.append(totalGenerations + "," + memoryPool.getLast().getFitness() + "," + memoryPool.getLast().size() +  "\n\n");
            FileWriter fw = new FileWriter(new File(System.getProperty("user.home") + "/Dropbox/cscdr_fitness_debug_" + data.relationName() + ".csv"), true);
            PrintWriter pw = new PrintWriter(fw);
            pw.print(statistics.toString());
            fw.close();
        }
    }
    
    private void insertRandomAntigens(Instances aAntigens){
        int newInsertions = (int)Math.round(newAbsPerGeneration*populationSize);
        for(int i=0; i < newInsertions; i++){
            memoryPool.add(generateNewAntibody(aAntigens));
        }
    }
    
    protected void clearAccumulatedHistory(){
    	for(CSCDRAntibodyPool abPool : memoryPool){
            for(CSCDRAntibody a : abPool.getAbList()){
                a.clearClassCounts();
            }
        }
    }
    
    protected void performCloningAndMutation(Instances aPartition){      
        Collections.sort(memoryPool, new comparatorAbPoolByFitness());
        int numClones;
        int size = memoryPool.size();
        // Uses the fitness of the abPool and not the one of the ab
        double fitnessMin = memoryPool.getFirst().fitness;
        double fitnessMaxMin = memoryPool.getLast().fitness - fitnessMin;
//        for(CSCDRAntibodyPool ab : memoryPool){
//            if (ab.getFitness() > fitnessMax) {
//                fitnessMax = ab.getFitness();
//            }
//        }
        
        for(int i=0; i < size; i++){
            numClones = (int)Math.round(getClonalScaleFactor()*size/(size - i));
                        
            for(int j=0; j < numClones; j++){    
                CSCDRAntibodyPool clone = new CSCDRAntibodyPool(memoryPool.get(i));
//                mutateAbPool(clone, aPartition);
                // Normaliza o fitness [0,1]
                double fitness = (clone.getFitness()-fitnessMin)/fitnessMaxMin;
                for(Antibody ab : clone.getAbList()){
                    mutateAntibody(ab, fitness, aPartition);
                }    
                // add to pool
    		memoryPool.add(clone);
            }
        }
    }  
    
    public void mutateAbPool(CSCDRAntibodyPool abPool, Instances aPartition){
        double fitMax = -1;
        for(int i=0; i < abPool.getAbList().size(); i++){
            if(abPool.getAbList().get(i).getFitness() > fitMax) {
                fitMax = abPool.getAbList().get(i).getFitness();
            }
        }
        for(int i=0; i < abPool.getAbList().size(); i++){
            double D = abPool.getAbList().get(i).getFitness()/fitMax;
            // ro = 3
            double ratio = Math.exp(-3*D); // e^(-3*i/n)  
            mutateAntibody(abPool.getAbList().get(i), ratio, aPartition);
        }
           
    }
    
    protected void mutateAntibody(Antibody aClone, double aMutationRate, Instances aPartition){
        double [][] minmax = affinityFunction.getMinMax();
        AttributeDistance[] attribs = affinityFunction.getDistanceMeasures();
        
        double [] data = aClone.getAttributes();
       
        for (int i = 0; i < data.length; i++){
            if(attribs[i].isClass()){
                continue;
            }
            else if(attribs[i].isNominal()){
                // Check mutation probability
                if(rand.nextDouble() <= aMutationRate)
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
    
    private void performPruning() {     
        Collections.sort(memoryPool, new comparatorAbPoolByFitness());
        int totalToRemove = memoryPool.size() - populationSize;
        for(int i=0; i < totalToRemove; i++) memoryPool.removeFirst();        
    }
    
    
    protected void calculateChromosomeFitness(CSCDRAntibodyPool abPool){
        double sumFitness = 0;
        for(CSCDRAntibody a : abPool.getAbList()){
            // check for a class switch
            if(a.canSwitchClass()){
                // perform a class switch
        	a.switchClasses();
        	
            }
                                   
            // calculate fitness
            a.calculateFitness(); 
            
//            sumFitness += a.getFitness();
            if(fitnessMode == 1){
                sumFitness += a.getFitness();
            }
            else{
                sumFitness += a.getTotalCorrect();
            }
        }
//        abPool.setFitness(sumFitness/abPool.size());
        if(fitnessMode == 1){
            abPool.setFitness(sumFitness/abPool.size());
        }
        else{
            abPool.setFitness(sumFitness);
        }
    }
    
    protected CSCDRAntibody selectBestMatchingUnit(Instance aInstance, LinkedList<CSCDRAntibody> abList){
        CSCDRAntibody bmu = null;
        // calculate affinity for population
        calculateAffinity(abList, aInstance);
        // sort by ascending numeric order - best affinity at zero
        Collections.sort(abList);
        // retrieve bmu
        bmu = abList.getFirst();
        return bmu;
    }
    
    protected void calculateAffinity(LinkedList<CSCDRAntibody> antibodies, Instance aInstance){
        double [] data = aInstance.toDoubleArray();
        
        for(CSCDRAntibody a : antibodies){
            double affinity = affinityFunction.calculateDistance(a.getAttributes(), data);
            a.setAffinity(affinity);
        }
    }

    private void exposeToAntigens(Instances data) {
        // expose each abPool to the partition
        for(CSCDRAntibodyPool abPool : memoryPool){ 
            for (int j = 0; j < data.numInstances(); j++){            
                // get a data instance
                Instance current = data.instance(j);
                // locate the best match
                CSCDRAntibody bmu = selectBestMatchingUnit(current, abPool.getAbList());
                // accumuate class counts
                bmu.updateClassCount(current);
            }        
            System.out.println(abPool.size());
            // calculate fitness for the abPool
            calculateChromosomeFitness(abPool);   
        }
        
    }

    private void testParameters(Instances data) {
        if(getChromosomeSize() <= 0 || getChromosomeSize() > data.numInstances()){
            throw new RuntimeException("The chromosome size must be greater than zero and less or equal to the number of training instances.");
    	}
        if(getPopulationSize() <= 0){
            throw new RuntimeException("The population size must be greater than zero.");
    	}
        if(getTotalGenerations() <= 0){
            throw new RuntimeException("The number of generations must be greater than zero.");
        }
        if(getClonalScaleFactor() <= 0){
            throw new RuntimeException("The clonal scale factor must be greater than zero.");
        }
        if(getNewAbsPerGeneration() < 0){
            throw new RuntimeException("The number of antibodies added by generation must be greater or equal to zero.");
        }
    }

    private void printMemoryCells() {
        System.out.print("\n\n============================================================================================\n");
        for(int i=0; i < memoryPool.size(); i++){
            System.out.print(memoryPool.get(i).toString() + "\n");
        }
        System.out.print("============================================================================================\n\n");
    }

    private LinkedList<CSCDRAntibody> selectBestAntibodyPool() {
        Collections.sort(memoryPool, new comparatorAbPoolByFitness());
        int i = memoryPool.size()-2;
        CSCDRAntibodyPool candidate = memoryPool.getLast();
        while(i >= 0){
            if(candidate.getFitness() == memoryPool.get(i).getFitness() && candidate.size() <= memoryPool.get(i).size()){
                candidate = memoryPool.get(i);
                i--;
            }
            else i = -1;
        }
        return candidate.getAbList();
    }
    
    /**
     * Interface para ordenação dos anticorpos por fitness (menor para maior)
     */
    public class comparatorAbPoolByFitness implements Comparator<CSCDRAntibodyPool> {
        public int compare(CSCDRAntibodyPool o1, CSCDRAntibodyPool o2) {
            if(o1.getFitness() < o2.getFitness()){
                return -1;
            }
            else if(o1.getFitness() > o2.getFitness()){
                return +1;
            }
            else
                return 0;
        }
    }
    
    
    private final static String [] PARAMETERS = {"C","P","G","S","B","D", "F"};
    
    private final static String [] DESCRIPTIONS ={"Chromosome size (nc).",
                                                  "Population size (np).",
                                                  "Number of generations (ng)",
                                                  "Random number generator seed (S).",
                                                  "Clonal scale factor (Beta).",
                                                  "Number of new antibodies added per generation (ns).",
                                                  "Fitness type utilized (F)"};
        
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
        // Chromosome size
        String aux = Utils.getOption(PARAMETERS[0], options);
        if (aux.length() != 0) {
            setChromosomeSize(Integer.parseInt(aux));
        }

        // Population size
        aux = Utils.getOption(PARAMETERS[1], options);
        if (aux.length() != 0) {
            setPopulationSize(Integer.parseInt(aux));
        }
        
        // Total generations
        aux = Utils.getOption(PARAMETERS[2], options);
        if (aux.length() != 0) {
            setTotalGenerations(Integer.parseInt(aux));
        }

        // Random number generator seed
        aux = Utils.getOption(PARAMETERS[3], options);
        if (aux.length() != 0) {
            setSeed(Long.parseLong(aux));
        }

        // Clonal scale factor
        aux = Utils.getOption(PARAMETERS[4], options);
        if (aux.length() != 0) {
            setClonalScaleFactor(Double.parseDouble(aux));
        }
        
        // Number of new cells added per generation
        aux = Utils.getOption(PARAMETERS[5], options);
        if (aux.length() != 0) {
            setNewAbsPerGeneration(Double.parseDouble(aux));
        }
        fitnessMode = getInteger(PARAMETERS[6], options);
    }
    
    
    public String[] getOptions(){
        LinkedList<String> list = new LinkedList<String>();

        String[] options = super.getOptions();
        for (int i = 0; i < options.length; i++){
            list.add(options[i]);
        }
        
        list.add("-" + PARAMETERS[0]);
        list.add(Integer.toString(getChromosomeSize()));    
        list.add("-" + PARAMETERS[1]);
        list.add(Integer.toString(getPopulationSize()));  
        list.add("-" + PARAMETERS[2]);
        list.add(Integer.toString(getTotalGenerations()));
        list.add("-" + PARAMETERS[3]);
        list.add(Long.toString(getSeed()));
        list.add("-" + PARAMETERS[4]);
        list.add(Double.toString(getClonalScaleFactor()));      
        list.add("-" + PARAMETERS[5]);
        list.add(Double.toString(getNewAbsPerGeneration()));  
        list.add("-" + PARAMETERS[6]);
        list.add(Integer.toString(fitnessMode));

        return list.toArray(new String[list.size()]);
    }
    
    public String chromosomeSizeTipText(){return DESCRIPTIONS[0];}    
    public String populationSizeTipText(){return DESCRIPTIONS[1];}
    public String totalGenerationsTipText(){return DESCRIPTIONS[2];}
    public String seedTipText(){return DESCRIPTIONS[3];}
    public String clonalScaleFactorTipText(){return DESCRIPTIONS[4];}  
    public String newAbsPerGenerationTipText(){return DESCRIPTIONS[5];}
    public String fitnessModeTipText(){return DESCRIPTIONS[6];}
	
    public int getChromosomeSize() {
        return chromosomeSize;
    }
    public void setChromosomeSize(int chromosomeSize) {
        this.chromosomeSize = chromosomeSize;
    }
    public int getPopulationSize() {
        return populationSize;
    }
    public void setPopulationSize(int populationSize) {
        this.populationSize = populationSize;
    }
    public int getTotalGenerations() {
        return totalGenerations;
    }
    public void setTotalGenerations(int totalGenerations) {
        this.totalGenerations = totalGenerations;
    }
    public long getSeed() {
        return seed;
    }
    public void setSeed(long seed) {
        this.seed = seed;
    }
    public double getClonalScaleFactor() {
        return clonalScaleFactor;
    }
    public void setClonalScaleFactor(double clonalScaleFactor) {
        this.clonalScaleFactor = clonalScaleFactor;
    }
    public double getNewAbsPerGeneration() {
        return newAbsPerGeneration;
    }
    public void setNewAbsPerGeneration(double newAbsPerGeneration) {
        this.newAbsPerGeneration = newAbsPerGeneration;
    }
    
    public void setFitnessMode(SelectedTag l) {
        if(l.getTags() == TAGS_FITNESS_MODE) {
            fitnessMode = l.getSelectedTag().getID();
        }
    }

    public SelectedTag getFitnessMode() {
        return new SelectedTag(fitnessMode, TAGS_FITNESS_MODE);
    }
   
    public static void main(String[] argv){
        try{
            System.out.println(Evaluation.evaluateModel(new CSCDR_v5_paper1fitness(), argv));
        }
        catch (Exception e){
            System.err.println(e.getMessage());
        }
    }
}
