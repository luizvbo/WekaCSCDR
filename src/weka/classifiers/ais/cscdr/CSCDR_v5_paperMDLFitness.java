package weka.classifiers.ais.cscdr;

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
import weka.classifiers.ais.cscdr.objects.CloneMachine;
import weka.classifiers.ais.cscdr.objects.ExhibitorOfAntigens;
import weka.core.*;

/**
 * Type: CSCDR_v5_paperMDLFitness <br>
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
public class CSCDR_v5_paperMDLFitness extends Classifier implements OptionHandler,weka.core.AdditionalMeasureProducer{
    public final static NumberFormat format = new DecimalFormat();
    // user paramters
//    private int antibodySize; // A
    private int populationSize; // P
    private int totalGenerations; // G
    private long seed; // S
    private double clonalScaleFactor; // B
    private double newAbsPerGeneration; // R	
    private int fitnessMode; // F
    private int kNN; // K
    private double alpha; // L
    
    private final static String[] measures = {"measureNumPrototypes"};
   
    private final static String [] PARAMETERS = {/*"A",*/"P","G","S","B","R", "F", "K","L"};
    
    private final static String [] DESCRIPTIONS ={//"Antibody maximum size (abs).",
                                                  "Population size (np).",
                                                  "Number of generations (ng)",
                                                  "Random number generator seed (S).",
                                                  "Clonal scale factor (Beta).",
                                                  "Number of new antibodies added per generation (ns).",
                                                  "Fitness type utilized (F).",
                                                  "Number of neighbors to consider (k).",
                                                  "Multplicative factor for MDL fitness: (1-a)*classificationMeasure+a*abSizeMeasure."
                                                };
    
    private final static Tag [] TAGS_FITNESS_MODE ={
//        new Tag(1, "MDL fitness"),
	new Tag(1, "Accuracy based fitness"),
        new Tag(2, "Recall based fitness"),
        new Tag(3, "F1 based fitness")
//        new Tag(5, "CSCDR based fitness")
    };
    
    private LinkedList<CSCDRAntibody> memoryCells;
    protected String trainingSummary;
    
    protected LinkedList<CSCDRAntibodyPool> memoryPool;
    protected ArrayList<CSCDRAntibodyPool> remainderPool;
    
    protected int remainderPoolSize;
//    protected LinkedList<Integer> antigenPool;
    protected Random rand;
    protected DistanceFunction affinityFunction;

    protected int partitionIndex;
    
    private int numberOfPrototypes;
    
//    private Normalize normalise;

   
    public CSCDR_v5_paperMDLFitness(){
        // set defaults
//        antibodySize = 35;
        populationSize = 50;
        totalGenerations = 50;
        seed = -1;
        clonalScaleFactor = 0.5;
        newAbsPerGeneration = 0.2;
        fitnessMode = 1;
        kNN = 3;
        alpha = 0.5;

        // TODO: should not be true by default
        m_Debug = false;
    }
    
    @Override
    public double classifyInstance(Instance aInstance){
    	// normalise vector
//	try {
//            normalise.input(aInstance);
//	} catch (Exception e) {
//            throw new RuntimeException("Unable to classify instance: "+e.getMessage(), e);
//	}
//        aInstance = normalise.output();
        
        // expose the system to the antigen
        return getKNNClassification(getKNN(aInstance, memoryCells), aInstance.numClasses());
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
        double stdev;
        double sum = 0.0;
        for (int i = 0; i < results.length; i++)
        {
            double diff = mean - results[i];
            sum += diff * diff;
        }
        stdev = Math.sqrt(sum / results.length);
        return stdev;
    }
        
    protected void algorithmPreperation(Instances aAntigens) throws Exception{
    	// prepare seed
        if(getSeed() < 0) {
            rand = new Random(System.currentTimeMillis());
        }
        else {
            rand = new Random(getSeed());
        }      
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
        
//        double gaus = rand.nextGaussian();
//        
//        // Normal distribution defines the size of the chromosome
//        int cSize = (int)Math.round((1-Math.abs(gaus % 3)/3)*antibodySize);
//        if (cSize == 0) cSize = 1;
        
        // doesn't allow abSize = 0
        int abSize = rand.nextInt(aAntigens.numInstances()-1)+1;
        
        for (int i = 0; i < abSize; i++){
            // clone the antigen as an antibody
            CSCDRAntibody clone = new CSCDRAntibody(aAntigens.instance(antigenPool.remove(0)));
            // add to pool
            ab.getAbList().add(clone);
        }
        return ab;
    }
    
    protected void initialiseAntibodyPool(Instances aAntigens) {
        memoryPool = new LinkedList<CSCDRAntibodyPool>();
        remainderPoolSize = (int)Math.round(populationSize*newAbsPerGeneration);
        remainderPool = new ArrayList<CSCDRAntibodyPool>(remainderPoolSize);
        for (int i = 0; i < populationSize-remainderPoolSize; i++) {
            memoryPool.add(generateNewAntibody(aAntigens));
        }
        for (int i = 0; i < remainderPoolSize; i++) {
            remainderPool.add(generateNewAntibody(aAntigens));
        }
    }
    
    public void buildClassifier(Instances data) throws Exception{    	
        
        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        data = new Instances(data);
        // Verifica se os parâmetros estão corretos
        testParameters(data);
        data.deleteWithMissingClass();
                  
//        normalise =  new Normalize();
//        normalise.setInputFormat(data);
//        data = Filter.useFilter(data, normalise);
        
        
        // prepare the algorithm
        algorithmPreperation(data);     
        // Debug mode
//        StringBuilder statistics = new StringBuilder();
        // Antibodies size control
//        ArrayList<Integer> abSizeControl = abSizeControlInitialization(data.numInstances());
        
        // train the system
        for (int generation = 0; generation < getTotalGenerations(); generation++){
            // randomise the dataset
//            data.randomize(rand);
              
            exposeToAntigens(data);       
            
            // perform pruning
            performPruning();
            
            if(m_Debug) {
                LinkedList<CSCDRAntibodyPool> allAntibodies = new LinkedList<CSCDRAntibodyPool>(memoryPool);
                allAntibodies.addAll(remainderPool);
                Collections.sort(allAntibodies, new comparatorAbPoolByFitness());
                
//                statistics.append(generation).append(",").append(memoryPool.getLast().getAccuracyOverTraining()).append(",").append(memoryPool.getLast().size()).append("\n");
                System.out.println(generation + "," + allAntibodies.getLast().getAccuracyOverTraining() + "," + allAntibodies.getLast().size());
                
            }
            
            performCloningAndMutation(data);
                    
            insertRandomAntigens(data);   
            
            clearAccumulatedHistory();
            
            // Debug
            if(generation % 10 == 0){
                System.out.println(generation);
            }
        }        
        exposeToAntigens(data);
   
        // Defines the memory pool as the best one from abPool
        CSCDRAntibodyPool bestAbPool = selectBestAntibodyPool();
        memoryCells = bestAbPool.getAbList();
        if(m_Debug){
            System.out.println(totalGenerations + "," + bestAbPool.getAccuracyOverTraining() + "," + bestAbPool.size());
//            statistics.append(totalGenerations).append(",").append(memoryPool.getLast().getFitness()).append(",").append(memoryPool.getLast().size()).append("\n");
//            System.out.print(statistics);
        }
        numberOfPrototypes = bestAbPool.size();
    }
    
    private void insertRandomAntigens(Instances aAntigens){
        remainderPool.clear();
        for(int i=0; i < remainderPoolSize; i++){
            remainderPool.add(generateNewAntibody(aAntigens));
        }
    }
    
    protected void clearAccumulatedHistory(){
    	for(CSCDRAntibodyPool abPool : memoryPool){
            for(CSCDRAntibody ab : abPool.getAbList()){
                ab.clearClassCounts();
            }
            abPool.clearFitness();
        }
    }
    
//    protected void performCloningAndMutation(Instances data){                           
//        int cores = Runtime.getRuntime().availableProcessors();
//        CloneMachine cloneMachines[] = new CloneMachine[cores];
//        LinkedList<CSCDRAntibodyPool>[] absPerCore = new LinkedList[cores];
//        
//        LinkedList<CSCDRAntibodyPool> allAntibodies = new LinkedList<CSCDRAntibodyPool>(memoryPool);
//        allAntibodies.addAll(remainderPool);
//                
//        Collections.sort(allAntibodies, new comparatorAbPoolByFitness());
//        
//        int size = allAntibodies.size();
//        // Uses the fitness of the abPool and not the one of the ab
//        double fitnessMin = allAntibodies.getFirst().fitness;
//        double fitnessMaxMin = allAntibodies.getLast().fitness - fitnessMin;
//        for(int j = 0; j < absPerCore.length; j++){
//            absPerCore[j] = new LinkedList<CSCDRAntibodyPool>();
//        }
//        int i = size, coreIndex = 0;
//        for(CSCDRAntibodyPool abPool : allAntibodies){
//            abPool.setNumberOfClones((int)Math.round(clonalScaleFactor*size/(i)));
//            absPerCore[coreIndex % cores].add(abPool);
//            coreIndex ++;           
//            i--;
//        }
//        
//        for (int k = 0; k < absPerCore.length; k++) {
//            cloneMachines[k] = new CloneMachine(absPerCore[k], data, this, fitnessMaxMin, fitnessMin);
//            cloneMachines[k].start();
//        }
//        for (int k = 0; k < absPerCore.length; k++) {
//            try {
//                cloneMachines[k].join();
//            }
//            catch (InterruptedException e) {
//                System.out.print("Join interrupted\n");
//            }
//        }
//    }  
    
     protected void performCloningAndMutation(Instances aPartition){           
        int numClones;
        LinkedList<CSCDRAntibodyPool> allAntibodies = new LinkedList<CSCDRAntibodyPool>(memoryPool);
        allAntibodies.addAll(remainderPool);
                
        Collections.sort(allAntibodies, new comparatorAbPoolByFitness());
        
        int size = allAntibodies.size();
        // Uses the fitness of the abPool and not the one of the ab
        double fitnessMin = allAntibodies.getFirst().fitness;
        double fitnessMaxMin = allAntibodies.getLast().fitness - fitnessMin;
//        double fitnessMax = allAntibodies.getLast().fitness;
//        for(CSCDRAntibodyPool ab : memoryPool){
//            if (ab.getFitness() > fitnessMax) {
//                fitnessMax = ab.getFitness();
//            }
//        }
        
        int i = size;
//        LinkedList<CSCDRAntibodyPool> clones = new LinkedList<CSCDRAntibodyPool>();
        for(CSCDRAntibodyPool abPool : allAntibodies){
            numClones = (int)Math.round(clonalScaleFactor*size/(i));
            i--;
            for(int j=0; j < numClones; j++){    
                CSCDRAntibodyPool clone = new CSCDRAntibodyPool(abPool);
//                mutateAbPool(clone, aPartition);
                // Normalize the fitness [0,1]
                double fitness = 0;
                if(fitnessMaxMin != 0){
                    fitness = (clone.getFitness()-fitnessMin)/fitnessMaxMin;
                }
                double ratio = Math.exp(-3*(fitness+0.1)); // e^(-3*i/n)  
//                double ratio = Math.exp(-3*memoryPool.get(i).getFitness()/fitnessMax);
                for(Antibody ab : clone.getAbList()){
                    mutateAntibody(ab, ratio, aPartition);
                }    
                // add to pool
    		memoryPool.add(clone);
//                clones.add(clone);
            }
        }
//        memoryPool.addAll(clones);
    }  
    
    public void cloningAndMutation(CSCDRAntibodyPool abPool, Instances antigens, double fitMaxMin, double fitMin){
        for(int j=0; j < abPool.getNumberOfClones(); j++){    
            CSCDRAntibodyPool clone = new CSCDRAntibodyPool(abPool);
            // Normalize the fitness [0,1]
            double fitness = 0;
            if(fitMaxMin != 0){
                fitness = (clone.getFitness()-fitMin)/fitMaxMin;
            }
            double ratio = Math.exp(-3*(fitness+0.1)); // e^(-3*i/n)  
//                double ratio = Math.exp(-3*memoryPool.get(i).getFitness()/fitnessMax);
            for(Antibody ab : clone.getAbList()){
                mutateAntibody(ab, ratio, antigens);
            }    
            // add to pool
            memoryPool.add(clone);
//                clones.add(clone);
        }
    }
    
//    public void mutateAbPool(CSCDRAntibodyPool abPool, Instances aPartition){
//        double fitMax = -1;
//        for(int i=0; i < abPool.getAbList().size(); i++){
//            if(abPool.getAbList().get(i).getFitness() > fitMax) {
//                fitMax = abPool.getAbList().get(i).getFitness();
//            }
//        }
//        for(int i=0; i < abPool.getAbList().size(); i++){
//            double D = abPool.getAbList().get(i).getFitness()/fitMax;
//            // ro = 3
//            double ratio = Math.exp(-3*D); // e^(-3*i/n)  
//            mutateAntibody(abPool.getAbList().get(i), ratio, aPartition);
//        }
//           
//    }
    
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
                if(rand.nextDouble() <= aMutationRate) {
                    data[i] = rand.nextInt(aPartition.attribute(i).numValues());
                }
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
        int totalToRemove = memoryPool.size() - (populationSize - remainderPoolSize);
        for(int i=0; i < totalToRemove; i++) {
            memoryPool.removeFirst();
        }        
    }
    
    
    private void calculateAntibodyFitness(CSCDRAntibodyPool abPool, int trainingSize) {
        for(CSCDRAntibody ab : abPool.getAbList()){
            // check for a class switch
            if(ab.canSwitchClass()){
                // perform a class switch
        	ab.switchClasses();	
            }
            // calculate fitness
//            a.calculateAccuracy();          
//            
//            sumFitness += a.getFitness();
        }
        double classFitness = 0;
        switch (fitnessMode){
            // Accuracy case
            case 1:   
                classFitness = getAccuracyFitness(abPool);
                if(m_Debug){
                    abPool.setAccuracyOverTraining(classFitness);
                }                
                break;
            // Recall case
            case 2:
//                classFitness = getAccuracyFitness(abPool);
                classFitness = getRecallFitness(abPool);
                if(m_Debug){
                    abPool.setAccuracyOverTraining(getAccuracyFitness(abPool));
                }
                break;
            // F1 case
            case 3:
                classFitness = getF1Fitness(abPool);
                if(m_Debug){
                    abPool.setAccuracyOverTraining(getAccuracyFitness(abPool));
                }
                break;
        }
        // size fitness -> smaller abs have bigger values
        double sizeFitness = (double)(trainingSize - abPool.size())/trainingSize;
        abPool.setFitness(alpha*classFitness+(1-alpha)*sizeFitness);
//        abPool.setFitness(sumFitness/abPool.size());
//        abPool.setFitness(sumFitness);
    }
    
    private double getAccuracyFitness(CSCDRAntibodyPool abPool) {
        int correct = 0, incorrect = 0; 
        for(CSCDRAntibody ab : abPool.getAbList()){
            int clasification = (int)ab.getClassification();
            for(int i = 0; i < ab.getClassCounts().length; i++){
                if(i == clasification){
                    correct += ab.getClassCounts()[i];
                }
                else{
                    incorrect += ab.getClassCounts()[i];
                }
            }
        }
        if(correct == 0) {
            return 0;
        }
        else {
            return (double)correct/(correct+incorrect);
        }
        
    }

    private double getRecallFitness(CSCDRAntibodyPool abPool) {
        int[] tp = null; 
        int[] fn = null;
        for(CSCDRAntibody ab : abPool.getAbList()){
            if(tp == null && fn == null){
                tp = new int[ab.getClassCounts().length];
                fn = new int[ab.getClassCounts().length];
            }
            int clasification = (int)ab.getClassification();
            for(int testClass = 0; testClass < ab.getClassCounts().length; testClass++){
                for(int i = 0; i < ab.getClassCounts().length; i++){
                    // Instancia positiva
                    if(testClass == clasification){
                        if(i == clasification){
                            tp[i] += ab.getClassCounts()[i];
                        }
                    }
                    // Instancia negativa
                    else{
                        if(i == testClass){
                            fn[i] += ab.getClassCounts()[i];
                        }
                    }
                }
            }
        }
        double recall = 0;
        for(int i = 0; i < tp.length; i++){
            if(tp[i] != 0) {
                recall += (double)tp[i]/(tp[i]+fn[i]);
            }
        }
        return recall/tp.length;
    }

    private double getF1Fitness(CSCDRAntibodyPool abPool) {
        int[] tp = null; 
        int[] fn = null;
        int[] fp = null;
        for(CSCDRAntibody ab : abPool.getAbList()){
            if(tp == null && fn == null){
                tp = new int[ab.getClassCounts().length];
                fn = new int[ab.getClassCounts().length];
                fp = new int[ab.getClassCounts().length];
            }
            int clasification = (int)ab.getClassification();
            for(int testClass = 0; testClass < ab.getClassCounts().length; testClass++){
                for(int i = 0; i < ab.getClassCounts().length; i++){
                    if(testClass == clasification){
                        if(i == clasification){
                            tp[i] += ab.getClassCounts()[i];
                        }
                        else{
                            fp[testClass] += ab.getClassCounts()[i];
                        }
                    }
                    else{
                        if(i == testClass){
                            fn[i] += ab.getClassCounts()[i];
                        }
                    }
                }
            }
        }
        double f1 = 0;
        for(int i = 0; i < tp.length; i++){
            // if tp = 0, f1 = 0
            if(tp[i] != 0){
                double precision = (double)tp[i]/(tp[i]+fp[i]);
                double recall = (double)tp[i]/(tp[i]+fn[i]);
                f1 += 2*precision*recall/(precision+recall);
            }
        }
        return f1/tp.length;
    }
    
    protected CSCDRAntibody selectBestMatchingUnit(Instance aInstance, LinkedList<CSCDRAntibody> abList){
        CSCDRAntibody bmu;
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
        LinkedList<CSCDRAntibodyPool> allAntibodies = new LinkedList<CSCDRAntibodyPool>(memoryPool);
        allAntibodies.addAll(remainderPool);
        // Split the memoryPool to be exposed by the number of cores of the CPU
        // and expose each part in a thread
        int cores = Runtime.getRuntime().availableProcessors();
        ExhibitorOfAntigens exsAg[] = new ExhibitorOfAntigens[cores];
        LinkedList<CSCDRAntibodyPool>[] absPerCore = new LinkedList[cores];
        // Initialize all lists
        for(int i = 0; i < absPerCore.length; i++){
            absPerCore[i] = new LinkedList<CSCDRAntibodyPool>();
        }
        int coreIndex = 0;
        for(CSCDRAntibodyPool abPool : allAntibodies){
            absPerCore[coreIndex % cores].add(abPool);
            coreIndex ++;
        }
        for (int i = 0; i < absPerCore.length; i++) {
            exsAg[i] = new ExhibitorOfAntigens(absPerCore[i], data, this);
            exsAg[i].start();
        }
        for (int i = 0; i < absPerCore.length; i++) {
            try {
                exsAg[i].join();
            }
            catch (InterruptedException e) {
                System.out.print("Join interrupted\n");
            }
        }
        
    }
    
    /**
     * Expose one antibody to the antigen and calculate its fitness
     * @param data Antigen (set of training instances)
     * @param abPool Antibody to be exposed
     */
    public void exposeAbPoolToAntigens(Instances data, CSCDRAntibodyPool abPool){
        
        for (int j = 0; j < data.numInstances(); j++){            
            // get a data instance
            Instance current = data.instance(j);
            // locate the neighbors match
            List<CSCDRAntibody> knn = getKNN(current, abPool.getAbList());
            double classification = getKNNClassification(knn, data.numClasses());
            // check which neighbors participate in the classification
            for(CSCDRAntibody ab : knn){
                if(ab.getClassification() == classification){
                    ab.updateClassCount(current);
                }
            }
//            CSCDRAntibody bmu = selectBestMatchingUnit(current, abPool.getAbList());
//            // accumuate class counts
//            bmu.updateClassCount(current);
        }        
        // calculate fitness for the abPool
        calculateAntibodyFitness(abPool, data.numInstances());   
    }

    private void testParameters(Instances data) {
//        if(getAntibodySize() <= 0 || getAntibodySize() > data.numInstances()){
//            throw new RuntimeException("The chromosome size must be greater than zero and less or equal to the number of training instances.");
//    	}
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

    private CSCDRAntibodyPool selectBestAntibodyPool() {
        LinkedList<CSCDRAntibodyPool> allAntibodies = new LinkedList<CSCDRAntibodyPool>(memoryPool);
        allAntibodies.addAll(remainderPool);
        
        Collections.sort(allAntibodies, new comparatorAbPoolByFitness());
//        int i = memoryPool.size()-2;
        return allAntibodies.getLast();
//        while(i >= 0){
//            if(candidate.getFitness() == memoryPool.get(i).getFitness() && candidate.size() <= memoryPool.get(i).size()){
//                candidate = memoryPool.get(i);
//                i--;
//            }
//            else {
//                i = -1;
//            }
//        }
    }

    /**
     * Return a List with the kNN (from abList) of an instance 
     * @param aInstance Instance in check
     * @param abList List of antibodies
     * @return List of kNN
     */
    private List<CSCDRAntibody> getKNN(Instance aInstance, LinkedList<CSCDRAntibody> abList){
        // calculate affinity for population
        calculateAffinity(abList, aInstance);
        // sort by ascending numeric order - best affinity at zero
        Collections.sort(abList);
        int toIndex = Math.min(kNN, abList.size());
        
//        toIndex = kNN;
        
        return abList.subList(0, toIndex);
    }
    
    private double getKNNClassification(List<CSCDRAntibody> knnList, int numClasses) {
        int[] nearesNeighbors = new int[numClasses];
        // count the number of NN per class         
        for(CSCDRAntibody ab : knnList){
            nearesNeighbors[(int)ab.getClassification()]++;
        }
        // check ties
        LinkedList<Integer> ties = new LinkedList<Integer>();
        int numberOfNN = 0;
        for(int i=0; i<nearesNeighbors.length; i++){
            if(nearesNeighbors[i] > numberOfNN){
                numberOfNN = nearesNeighbors[i];
                ties.clear();
                ties.add(i);
            }
            else if(nearesNeighbors[i] == numberOfNN){
                ties.add(i);
            }
        }
//        if(numberOfTies == 0) {
//            return classification;
//        }
//        else{
//            int randomClass = rand.nextInt(numberOfTies+1);
//            int classClounter = 0;
//            for(int i = 0; i < ; ){
//                
//            }
//        }
        // retrieve bmu
//        bmu = abList.getFirst();
//        return bmu;
        if(ties.size() == 1){
            // Nao ha conflito. Retorna a unica classe de ties
            return ties.get(0);
        }
        else{
            // A classe e definida pela instancia mais prooxima, pertencente as
            //classes em conflito
            for(CSCDRAntibody ab : knnList){
                int classificationTest = (int)ab.getClassification();
                if(ties.contains(classificationTest)){
                    return classificationTest;
                }
            }
        }
        return 0;
    }

    public Enumeration enumerateMeasures() {
        List<String> measuresArrayList = Arrays.asList(measures);
        return Collections.enumeration(measuresArrayList);
    }

    public double getMeasure(String measureName) {
        if(measureName.equals(measures[0])){
            return numberOfPrototypes;
        }
        else{
            return 0;
        }
    }


//    private ArrayList<Integer> abSizeControlInitialization(int numInstances) {
//        ArrayList<Integer> abSizeControl = new ArrayList<Integer>(numInstances);
//        for(int i = 0; i < numInstances; i++){
//            abSizeControl.add(i);
//        }
//        Collections.sort(abs);
//    }
    
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
            else {
                return 0;
            }
        }
    }
    
    
        
    @Override
    public String toString(){
        StringBuilder buffer = new StringBuilder(1000);
        buffer.append("Clonal Selection Classification with  Data Reduction (CSCDR) v1.0.\n");
        
        if(trainingSummary != null){
        	buffer.append("\n");
        	buffer.append(trainingSummary);
        }
        
        return buffer.toString();
    }

    public String globalInfo(){
        StringBuilder buffer = new StringBuilder(1000);
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

    @Override
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

    /**
     * Try to get a parameter (double) from the input arguments. If it doesn't find, return the default parameter.
     * @param param Parameter flag
     * @param options List of input parameters
     * @param defaultParameter Default parameter
     * @return The new value if it is find or the default value otherwise
     * @throws Exception Error on paramter format
     */
    protected double getDouble(String param, String[] options, double defaultParameter) throws Exception{
        String value = Utils.getOption(param.charAt(0), options);
        if (value.length() != 0){
            try{
                return Double.parseDouble(value);
            }catch(NumberFormatException ex){
                throw new NumberFormatException("Parameter format error");
            }
        }
        else{
            return defaultParameter;
        }
    }

    /**
     * Try to get a parameter (integer) from the input arguments. If it doesn't find, return the default parameter.
     * @param param Parameter flag
     * @param options List of input parameters
     * @param defaultParameter Default parameter
     * @return The new value if it is find or the default value otherwise
     * @throws Exception Error on paramter format
     */
    protected int getInteger(String param, String[] options, int defaultParameter) throws Exception{
        String value = Utils.getOption(param.charAt(0), options);
        if (value.length() != 0){
            try{
                return Integer.parseInt(value);
            }catch(NumberFormatException ex){
                throw new NumberFormatException("Parameter format error");
            }
        }
        else{
            return defaultParameter;
        }
    }

    /**
     * Try to get a parameter (long) from the input arguments. If it doesn't find, return the default parameter.
     * @param param Parameter flag
     * @param options List of input parameters
     * @param defaultParameter Default parameter
     * @return The new value if it is find or the default value otherwise
     * @throws Exception Error on paramter format
     */
    protected long getLong(String param, String[] options, long defaultParameter) throws Exception{
        String value = Utils.getOption(param.charAt(0), options);
        if (value.length() != 0){
            try{
                return Long.parseLong(value);
            }catch(NumberFormatException ex){
                throw new NumberFormatException("Parameter format error");
            }
        }
        else{
            return defaultParameter;
        }
    }


    @Override
    public void setOptions(String[] options) throws Exception{
//        antibodySize = getInteger(PARAMETERS[0], options);
        m_Debug = Utils.getFlag('D', options);
        populationSize = getInteger(PARAMETERS[0], options, populationSize);
        totalGenerations = getInteger(PARAMETERS[1], options, totalGenerations);
        seed = getLong(PARAMETERS[2], options, seed);
        clonalScaleFactor = getDouble(PARAMETERS[3], options, clonalScaleFactor);    
        newAbsPerGeneration = getDouble(PARAMETERS[4], options, newAbsPerGeneration);        
        fitnessMode = getInteger(PARAMETERS[5], options, fitnessMode);
        kNN = getInteger(PARAMETERS[6], options, kNN);
        alpha = getDouble(PARAMETERS[7], options, alpha);
    }
    
    
    @Override
    public String[] getOptions(){
        LinkedList<String> list = new LinkedList<String>();

        String[] options = super.getOptions();
        for (int i = 0; i < options.length; i++){
            list.add(options[i]);
        }
        
//        list.add("-" + PARAMETERS[0]);
//        list.add(Integer.toString(getAntibodySize()));    
        list.add("-" + PARAMETERS[0]);
        list.add(Integer.toString(getPopulationSize()));  
        list.add("-" + PARAMETERS[1]);
        list.add(Integer.toString(getTotalGenerations()));
        list.add("-" + PARAMETERS[2]);
        list.add(Long.toString(getSeed()));
        list.add("-" + PARAMETERS[3]);
        list.add(Double.toString(getClonalScaleFactor()));      
        list.add("-" + PARAMETERS[4]);
        list.add(Double.toString(getNewAbsPerGeneration()));  
        list.add("-" + PARAMETERS[5]);
        list.add(Integer.toString(fitnessMode));  
        list.add("-" + PARAMETERS[6]);
        list.add(Integer.toString(getKNN()));  
        list.add("-" + PARAMETERS[7]);
        list.add(Double.toString(getAlpha()));

        return list.toArray(new String[list.size()]);
    }
    
//    public String antibodySizeTipText(){return DESCRIPTIONS[0];}    
    public String populationSizeTipText(){return DESCRIPTIONS[0];}
    public String totalGenerationsTipText(){return DESCRIPTIONS[1];}
    public String seedTipText(){return DESCRIPTIONS[2];}
    public String clonalScaleFactorTipText(){return DESCRIPTIONS[3];}  
    public String newAbsPerGenerationTipText(){return DESCRIPTIONS[4];}
    public String fitnessModeTipText(){return DESCRIPTIONS[5];}
    public String kNNTipText(){return DESCRIPTIONS[6];}
    public String alphaTipText(){return DESCRIPTIONS[7];}
    
   
//    public int getAntibodySize() {
//        return antibodySize; 
//    }
//    public void setAntibodySize(int antibodySize) {
//        this.antibodySize = antibodySize;
//    }
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
    public int getKNN() {
        return kNN;
    }
    public void setKNN(int kNN) {
        this.kNN = kNN;
    }

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alfa) {
        this.alpha = alfa;
    }
   
    public static void main(String[] argv){
        try{
            System.out.println(Evaluation.evaluateModel(new CSCDR_v5_paperMDLFitness(), argv));
        }
        catch (Exception e){
            System.err.println(e.getMessage());
        }
    }
}
