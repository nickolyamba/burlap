package edu.brown.cs.burlap.assignment4;

import burlap.behavior.learningrate.ExponentialDecayLR;
import burlap.behavior.learningrate.LearningRate;
import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.QProvider;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.debugtools.MyTimer;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.*;
import java.util.List;

/**
 * @author Nikolay Goncharenko.
 */
public class MyGridWorld {

	private GridWorldDomain gw;
    private OOSADomain domain;
    private TerminalFunction tf;
    private StateConditionTest goalCondition;
    private State initialState;
    private HashableStateFactory hashingFactory;
    private SimulatedEnvironment env;

    private  int [][] map;
    private static DecimalFormat dfTime = new DecimalFormat("0.0000");

	// http://burlap.cs.brown.edu/tutorials/bpl/p2.html
	public MyGridWorld(int [][] map, int[] agentXY, int [] goalXY){
        // http://burlap.cs.brown.edu/doc/burlap/domain/singleagent/gridworld/GridWorldDomain.html
        double stochasticity = 0.9;
        this.map = map;
		gw = new GridWorldDomain(map);
        gw.setProbSucceedTransitionDynamics(stochasticity); //stochastic transitions with 0.8 success rate
		//gw.setMapToFourRooms();
		tf = new GridWorldTerminalFunction(goalXY[0], goalXY[1]);
		gw.setTf(tf);

		goalCondition = new TFGoalCondition(tf);

        /*
        // Reward Function
        GridWorldRewardFunction rf = new GridWorldRewardFunction(5, 5);
        rf.setReward(goalXY[0], goalXY[1], 50.0);
        gw.setRf(rf);
        */
        domain = gw.generateDomain();

		initialState = new GridWorldState(new GridAgent(agentXY[0], agentXY[1]), new GridLocation(goalXY[0], goalXY[1], "goal"));
		hashingFactory = new SimpleHashableStateFactory();

		//((FactoredModel)domain.getModel()).setRf(new GoalBasedRF(this.goalCondition, 1.0, -1));
		env = new SimulatedEnvironment(domain, initialState);

        // GridWorldRewardFunction

		//VisualActionObserver observer = new VisualActionObserver(domain, GridWorldVisualizer.getVisualizer(gw.getMap()));
		//observer.initGUI();
		//env.addObservers(observer);
	}


	public void visualize(String outputpath){
		Visualizer v = GridWorldVisualizer.getVisualizer(gw.getMap());
		new EpisodeSequenceVisualizer(v, domain, outputpath);
	}



    // http://www.programcreek.com/java-api-examples/index.php?source_dir=burlap-master/src/burlap/behavior/
    // singleagent/planning/stochastic/policyiteration/PolicyIteration.java
	private List<Double> valueIteration(String outputPath){
        System.out.println("\nValue Iteration: \n---------------------------------------\n");
        double discountRate = 0.9, termDelta= 0.001;
        int maxIterations = 13;

        // We've chossen for VI to terminate when either the changes in the value function
        // are no longer than 0.001, or 100 iterations
        Planner planner = new ValueIteration(domain, discountRate, hashingFactory, termDelta, maxIterations);

        // run algorithm
        double start = System.nanoTime();
        Policy p = planner.planFromState(initialState);
        double end = System.nanoTime();
        double compTime = end - start;
        compTime /= Math.pow(10,9);

        Episode e = PolicyUtils.rollout(p, initialState, domain.getModel());
        e.write(outputPath + "vi");

        //writeEpisodeToCSV(PolicyUtils.rollout(p, initialState, domain.getModel()), maxIterations, compTime, "VI_small");

        String title = "VI. Time:%4$.4f  Discount: %1$.2f  Termination = %2$.3f  MaxIters = %3$d";
        String output = String.format(title, discountRate, termDelta, maxIterations, compTime);

        List<Double> allValues = simpleValueFunctionVis((ValueFunction)planner, p, output);
        System.out.println(output);
        //manualValueFunctionVis((ValueFunction)planner, p);

        return allValues;
	}

    public List<Double> testValueIteration(String filename){
        System.out.println("\nValue Iteration: \n---------------------------------------\n");
        double discountRate = 0.9, termDelta= 0.001, compTime=0.0, start, end, numActions=0.0;
        int maxIterations = 0;
        final int MAX_ITER = 15, TRIALS = 10;

        Planner planner = null;
        Policy p = null;
        Episode e = null;

        // hashmap to store time and number of steps for each iteration
        Map<Integer, List<List<Double>> > map = initResultMap(MAX_ITER);

        for(int j = 1; j <= TRIALS; j++)
        {
            for(int i = 1; i <= 15; i++)
            {
                maxIterations = i;
                // We've chossen for VI to terminate when either the changes in the value function
                // are no longer than 0.001, or 100 iterations
                planner = new ValueIteration(domain, discountRate, hashingFactory, termDelta, maxIterations);

                // run algorithm
                start = System.nanoTime();
                p = planner.planFromState(initialState);
                end = System.nanoTime();

                compTime = end - start;
                compTime /= Math.pow(10,9);

                e = PolicyUtils.rollout(p, initialState, domain.getModel());

                numActions = (double)e.numActions();
                // Store results in the map
                map.get(i).get(0).add(compTime);
                map.get(i).get(1).add(numActions);

                System.out.println(String.format("Time: %1$.4f,  Steps: %2$.1f\n\n", compTime, numActions));
                // write to CSV
                //writeEpisodeToCSV(PolicyUtils.rollout(p, initialState, domain.getModel()), maxIterations, compTime, filename);
            }//for i run VI with different number of iterations
        }// for j

        // Get avg for each iteration and save in .csv
        Map<Integer, List<Double>> mapAverage = mapAverage(map, MAX_ITER);
        writeToCSVAverage(mapAverage, filename);

        // show result of the last iter = MAX_ITER
        e.write("output/" + "vi");
        String title = "VI. Time:%4$.4f  Discount: %1$.2f  Termination = %2$.3f  MaxIters = %3$d";
        String output = String.format(title, discountRate, termDelta, maxIterations, compTime);
        List<Double> allValues = simpleValueFunctionVis((ValueFunction)planner, p, output);
        System.out.println(output);

        return allValues;
    }

    // http://burlap.cs.brown.edu/doc/burlap/behavior/singleagent/planning/stochastic/policyiteration/PolicyIteration.html
    public List<Double>  policyIteration(String outputPath){
        double discountRate = 0.9, evalDelta = 0.001, PIDelta = 0.1;
        int maxEvaluationIterations=2, maxPolicyIterations=100;

        System.out.println("\nPolicy Iteration: \n---------------------------------------\n");
        // We've chossen for VI to terminate when either the changes in the value function
        // are no longer than 0.001, or 100 iterations
        Planner planner = new PolicyIteration(domain, discountRate, hashingFactory, PIDelta, evalDelta,
                                                maxEvaluationIterations, maxPolicyIterations);
        double start = System.nanoTime();
        Policy p = planner.planFromState(initialState);
        double end = System.nanoTime();
        double compTime = end - start;
        compTime /= Math.pow(10,9);

        PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "pi");

        int policyIters = ((PolicyIteration)planner).getTotalPolicyIterations();
        int valueIters = ((PolicyIteration)planner).getTotalValueIterations();
        String title = "PI. Time:%5$.4f  Discount: %1$.2f  Termination = %2$.3f  PolicyIters = %3$d  ValueIters = %4$d";
        String output = String.format(title, discountRate, evalDelta, policyIters, valueIters, compTime);

        List<Double> allValues = simpleValueFunctionVis((ValueFunction)planner, p, output);
        System.out.println(output);

        return allValues;
    }

    public List<Double>  testPolicyIteration(String filename){
        System.out.println("\nPolicy Iteration: \n---------------------------------------\n");
        double discountRate = 0.9, evalDelta = 0.0000000001, PIDelta = 0.000000001,
                start, end, compTime=0.0, numActions=0.0;
        int maxEvaluationIterations=2, maxPolicyIters;
        final int MAX_ITER = 15, TRIALS = 10;

        // We've chossen for VI to terminate when either the changes in the value function
        // are no longer than 0.001, or 100 iterations
        Planner planner = null;
        Policy p = null;
        Episode e = null;

        // hashmap to store time and number of steps for each iteration
        Map<Integer, List<List<Double>> > map = initResultMap(MAX_ITER);

        // 10 iterations to produce an average result
        for(int j = 1; j <= TRIALS; j++)
        {
            for(int i = 1; i <= 15; i++)
            {
                maxPolicyIters = i;
                planner = new PolicyIteration(domain, discountRate, hashingFactory, PIDelta, evalDelta,
                        maxEvaluationIterations, maxPolicyIters);

                start = System.nanoTime();
                p = planner.planFromState(initialState);
                end = System.nanoTime();

                compTime = end - start;
                compTime /= Math.pow(10,9);

                e = PolicyUtils.rollout(p, initialState, domain.getModel());
                //e.write("output/" + "pi");

                numActions = (double)e.numActions();
                // Store results in the map
                map.get(i).get(0).add(compTime);
                map.get(i).get(1).add(numActions);

                System.out.println(String.format("Time: %1$.4f,  Steps: %2$.1f\n\n", compTime, numActions));
                // write to CSV
                //writeEpisodeToCSV(PolicyUtils.rollout(p, initialState, domain.getModel()), maxPolicyIters, compTime, filename);
            }//for i  run PI with different number of iterations
        }//for j number of trials

        // Get avg for each iteration and save in .csv
        Map<Integer, List<Double>> mapAverage = mapAverage(map, MAX_ITER);
        writeToCSVAverage(mapAverage, filename);

        int policyIters = ((PolicyIteration)planner).getTotalPolicyIterations();
        int valueIters = ((PolicyIteration)planner).getTotalValueIterations();

        // show result of the last iter = MAX_ITER
        e.write("output/" + "pi");
        String title = "PI. Time:%5$.4f  Discount: %1$.2f  Termination = %2$.3f  PolicyIters = %3$d  ValueIters = %4$d";
        String output = String.format(title, discountRate, evalDelta, policyIters, valueIters, compTime);
        List<Double> allValues = simpleValueFunctionVis((ValueFunction)planner, p, output);
        System.out.println(output);

        return allValues;
    }

    /*                  SMALL
    * ExponentialDecayLR(0.9, 0.99998, 0.0001);
    * for(i <= 20000) if(i % 1000 == 0)epsilon -= 0.05;
    * */
	public List<Double> qLearning(String outputPath){
        System.out.println("\nQ-Learning: \n---------------------------------------\n");

        final int NUM_ITER = 100;
        int numActions = 0, mod_factor = NUM_ITER/10;
        double discountRate = 0.9, qInitial = 0.0, learningRate_const = 0.9,
                iterTime=0.0, totalTime, epsilon=0.9, epsilonFactor = mod_factor*epsilon/(NUM_ITER*0.9);

        // domain, a discount factor, a HashableStateFactory, an initial value for the Q-values,
        // and a learning rate (which for a deterministic domain, 1.0 is a good choice)
        // http://burlap.cs.brown.edu/doc/burlap/behavior/singleagent/learning/tdmethods/QLearning.html

        //learningRate = new SoftTimeInverseDecayLR(1.0, 5, 0.0001);
        LearningAgent agent = new QLearning(domain, discountRate, hashingFactory, qInitial, learningRate_const);
        LearningRate learningRate = new ExponentialDecayLR(0.6, 0.9998, 0.001);
        EpsilonGreedy epsilonObj = new EpsilonGreedy((QProvider) agent, epsilon);
        ((QLearning)agent).setLearningPolicy(epsilonObj);
        ((QLearning)agent).setLearningRateFunction(learningRate);

        Episode e = null;
        MyTimer timer = new MyTimer();
		for(int i = 1; i <= NUM_ITER; i++){
            timer.start();
            e = agent.runLearningEpisode(env);
            timer.stop();
            //iterTime = timer.getTime();
            //totalTime = timer.getTotalTime();
            //numActions = e.numActions();

            //writeEpisodeToCSV(e, i, totalTime, "qLearning_small_100");

			if(i % mod_factor == 0){
				epsilon -= epsilonFactor;
				if(epsilon < 0) epsilon = 0.0;
				epsilonObj = new EpsilonGreedy((QProvider) agent, epsilon);
				((QLearning)agent).setLearningPolicy(epsilonObj);
                System.out.println(String.format("epsilon: %1.2f", epsilon));
			}

			//e.write(outputPath + "ql_" + i);
			//System.out.println(i+1 + ": " + e.maxTimeStep() + "\t" + iterTime + " s");

			//reset to an initial state from its current state, which may be a terminal state
			env.resetEnvironment();
		}

        totalTime = timer.getTotalTime();

        // output last iter
        e.write(outputPath + "ql");

        String title = "Q-Learning. Iterations: %5$d  Time: %3$.4f  Discount: %1$.2f  StepsToGoal = %2$d  Epsilon = %4$.1f";
        String output = String.format(title, discountRate, e.numActions(), totalTime, epsilon, NUM_ITER);

        List<Double> allValues = simpleValueFunctionVis((ValueFunction)agent, new GreedyQPolicy((QProvider) agent), output);
        System.out.println(output);

		//agent = new QLearning(domain, discountRate, hashingFactory, 1.0, 0.1);
		//experimentAndPlotter(agent);

        return allValues;
	}

    // Build Map to store results of each iteration and then use it to calculate averages
    private Map<Integer, List<List<Double>> > initResultMap(int NUM_ITER){
        Map<Integer, List<List<Double>> > map = new HashMap<>();
        for(int i = 1; i <= NUM_ITER; i++){
            List<List<Double>> list = new ArrayList<>();
            List<Double> listTime = new ArrayList<>();
            List<Double> listSteps = new ArrayList<>();
            list.add(listTime); list.add(listSteps);
            map.put(i, list);
        }//for
        return map;
    }//initResultMap()

    private Map<Integer, List<Double>> mapAverage(Map<Integer, List<List<Double>> > avgMap, int NUM_ITER){
        Map<Integer, List<Double>> mapAverage = new HashMap<>();
        for(int i = 1; i <= NUM_ITER; i++){
            double avgTime = avgMap.get(i).get(0).stream().mapToDouble((x) -> x).average().orElse(0.0);
            double avgSteps = avgMap.get(i).get(1).stream().filter(Objects::nonNull).mapToDouble((x) -> x).average().orElse(0.0);
            //double avgSteps = avgMap.get(i).get(1).stream().filter(Objects::nonNull).mapToDouble((x) -> x).average().getAsDouble();
            List<Double> avgList = new ArrayList<>(2);
            avgList.add(avgTime); avgList.add(avgSteps);
            mapAverage.put(i, avgList);
        }//for
        return mapAverage;
    }//mapAverage()

    public List<Double> testQLearning(String filename){
        System.out.println("\nQ-Learning: \n---------------------------------------\n");

        final int NUM_ITER = 100;
        int numActions = 0, mod_factor = NUM_ITER/10;
        double discountRate = 0.9, qInitial = 0.0, learningRate_const = 0.9,
                iterTime=0.0, totalTime = 0.0, epsilon=0.9, epsilonFactor = mod_factor*epsilon/(NUM_ITER*0.9);

        // hashmap to store time and number of steps for each iteration
        Map<Integer, List<List<Double>> > map = initResultMap(NUM_ITER);

        Episode e = null;
        LearningAgent agent = null;
        LearningRate learningRate;

        for(int j = 1; j <= 10; j++)
        {
            epsilon=0.5;
            agent = new QLearning(domain, discountRate, hashingFactory, qInitial, learningRate_const);
            learningRate = new ExponentialDecayLR(0.8, 0.9998, 0.001);
            EpsilonGreedy epsilonObj = new EpsilonGreedy((QProvider) agent, epsilon);
            ((QLearning)agent).setLearningPolicy(epsilonObj);
            ((QLearning)agent).setLearningRateFunction(learningRate);

            MyTimer timer = new MyTimer();
            for(int i = 1; i <= NUM_ITER; i++){
                timer.start();
                e = agent.runLearningEpisode(env);
                timer.stop();
                //iterTime = timer.getTime();
                totalTime = timer.getTotalTime();

                //writeEpisodeToCSV(e, i, totalTime, "qLearning_small_100");

                // Store results in the map
                map.get(i).get(0).add(totalTime);
                map.get(i).get(1).add((double)e.numActions());

                if(i % mod_factor == 0){
                    epsilon -= epsilonFactor;
                    if(epsilon < 0) epsilon = 0.0;
                    epsilonObj = new EpsilonGreedy((QProvider) agent, epsilon);
                    ((QLearning)agent).setLearningPolicy(epsilonObj);
                    System.out.println(String.format("epsilon: %1.2f", epsilon));
                }

                //e.write(outputPath + "ql_" + i);
                //System.out.println(i+1 + ": " + e.maxTimeStep() + "\t" + iterTime + " s");

                //reset to an initial state from its current state, which may be a terminal state
                env.resetEnvironment();
            }//for i

            totalTime = timer.getTotalTime();
            /*
            String title = "Q-Learning. Iterations: %5$d  Time: %3$.4f  Discount: %1$.2f  StepsToGoal = %2$d  Epsilon = %4$.1f";
            String output = String.format(title, discountRate, e.numActions(), totalTime, epsilon, NUM_ITER);
            List<Double> allValues = simpleValueFunctionVis((ValueFunction)agent, new GreedyQPolicy((QProvider) agent), output);
            System.out.println(output);*/
        }//for j

        // Get avg for each iteration and save in .csv
        Map<Integer, List<Double>> mapAverage = mapAverage(map, NUM_ITER);
        writeToCSVAverage(mapAverage, filename);

        // output last iter
        e.write("output/" + "ql");

        // show result
        String title = "Q-Learning. Iterations: %5$d  Time: %3$.4f  Discount: %1$.2f  StepsToGoal = %2$d  Epsilon = %4$.1f";
        String output = String.format(title, discountRate, e.numActions(), totalTime, epsilon, NUM_ITER);
        List<Double> allValues = simpleValueFunctionVis((ValueFunction)agent, new GreedyQPolicy((QProvider) agent), output);
        System.out.println(output);

        //agent = new QLearning(domain, discountRate, hashingFactory, 1.0, 0.1);
        //experimentAndPlotter(agent);

        return allValues;
    }

    public void writeToCSVAverage(Map<Integer, List<Double>> mapAverage, String filename){
        StringBuffer result = new StringBuffer();
        String path = String.format("results/%1s.csv", filename);
        String headerLine = "Iteration#,Time,Actions,\n";

        int numIter = mapAverage.size();
        for(int i = 1; i <= numIter; i++){
            result.append(i); result.append(",");
            result.append(dfTime.format(mapAverage.get(i).get(0))); result.append(",");
            result.append(Math.round(mapAverage.get(i).get(1))); result.append("\n");
        }

        writeToCSV(path, headerLine, result);
        System.out.println(result);
    }

    public void writeEpisodeToCSV(Episode e, int iteration, double time, String filename){
        StringBuffer result = new StringBuffer();
        String path = String.format("results/%1s.csv", filename);
        String headerLine = "Iteration#,Time,Actions,Rewards\n";
        int cumReward = e.rewardSequence.size();

        result.append(iteration); result.append(",");
        result.append(dfTime.format(time)); result.append(",");
        result.append(e.numActions()); result.append(",");
        result.append(cumReward); result.append("\n");

        writeToCSV(path, headerLine, result);
        System.out.println(result);
    }


	public List<Double>  simpleValueFunctionVis(ValueFunction valueFunction, Policy p, String title){

		List<State> allStates = StateReachability.getReachableStates(initialState, domain, hashingFactory);
		ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(allStates, map.length, map[0].length, valueFunction, p);
        List<Double> allValues = new ArrayList<>();
        for(State state : allStates)
            allValues.add(valueFunction.value(state));
        gui.setTitle(title);
		gui.initGUI();

        //saveWindow(gui.getContentPane());

        return allValues;
	}

    public void saveWindow(Container dPanel)
    {
        BufferedImage bImg = new BufferedImage(dPanel.getWidth(), dPanel.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D cg = bImg.createGraphics();
        dPanel.paintAll(cg);
        try {
            if (ImageIO.write(bImg, "jpeg", new File("./output_image.jpeg")))
            {
                System.out.println("-- saved");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


	public void experimentAndPlotter(final LearningAgent agent){

		//different reward function for more structured performance plots

		//((FactoredModel)domain.getModel()).setRf(new GoalBasedRF(this.goalCondition, 50.0, -0.5));

		/**
		 * Create factories for Q-learning agent and SARSA agent to compare
		 */
		LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

			public String getAgentName() {
				return "Q-Learning";
			}
			public LearningAgent generateAgent() {
				return new QLearning(domain, 0.9, hashingFactory, 0.3, 0.1);//agent;
			}
		};
		/*
		LearningAgentFactory sarsaLearningFactory = new LearningAgentFactory() {

			public String getAgentName() {
				return "SARSA";
			}


			public LearningAgent generateAgent() {
				return new SarsaLam(domain, 0.99, hashingFactory, 0.0, 0.1, 1.);
			}
		};*/

		LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env, 2, 5000, qLearningFactory);

        // plot's width and height, the number of columns of plots, and the maximum window height.
        // In this case, plots are set to be 500x200, with two columns of plots
        exp.setUpPlottingConfiguration(500, 250, 2, 1000,
				TrialMode.MOST_RECENT_AND_AVERAGE,
				PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE,
				PerformanceMetric.AVERAGE_EPISODE_REWARD);

		exp.startExperiment();
		exp.writeStepAndEpisodeDataToCSV("expData");

	}

    private static void runTests(int [][] map, int[] agentXY, int [] goalXY){
        MyGridWorld gridWorld = new MyGridWorld(map, agentXY, goalXY);
        String outputPath = "output/";

        //example.sarsaLearningExample(outputPath);
        //List<Double> PIValues = gridWorld.policyIteration(outputPath);
        //List<Double> VIValues = gridWorld.valueIteration(outputPath);
        //List<Double> QValues = gridWorld.testQLearning("Q_small");

        //gridWorld.testPolicyIteration("TI_small");

        List<Double> VIValues = gridWorld.testValueIteration("VI_small");
        //List<Double> QValues = gridWorld.qLearning(outputPath);

        //gridWorld.visualize(outputPath);

        /*
        double delta = 0;
        //example.experimentAndPlotter();
        for(int i = 0; i < VIValues.size(); i++)
            delta += Math.abs(VIValues.get(i) - QValues.get(i));

        System.out.println("Difference: " + delta);
        */
    }

	public static void main(String[] args) {
		int IS_SMALL = 1;

		int [][] map_small = new int[][]{
				{0,0,0,1,0},
				{0,0,0,0,0},
				{1,1,1,1,0},
				{0,0,0,0,0},
				{0,0,0,0,0}
		};
         int [][] map_large = new int[][]{
			{0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,1,0,1,0,0,0,0,1,1,1,1,1,0,1,1},
			{0,0,0,0,0,1,0,1,1,1,1,0,0,0,1,0,0,0,0,0},
			{0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0},
			{1,0,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
        };

		if(IS_SMALL == 1) {
			int[] agentXY = new int[]{0, 0};
			int[] goalXY = new int[]{4, 0};
			runTests(map_small, agentXY, goalXY);
		}
		else {
			int[] agentXY = new int[]{0, 4};
			int[] goalXY = new int[]{15, 15};
			runTests(map_large, agentXY, goalXY);
		}

	}//main()

    private static void writeToCSV(String filename, String headerLine, StringBuffer result) {
        try{
            File file = new File(filename);
            FileWriter fileWriter;

            //if file doesn't exists, then create it and add headerLine
            if(!file.exists()){
                // Create parent dir if doesn't exist
                File parent = file.getParentFile();
                if(!parent.exists() && !parent.mkdirs())
                    throw new IllegalStateException("Couldn't create dir: " + parent);

                // Create file and add header
                file.createNewFile();
                fileWriter = new FileWriter(filename);
                BufferedWriter bufferWritter = new BufferedWriter(fileWriter);
                bufferWritter.write(headerLine);
                bufferWritter.close();
            }

            // 'true' in FileWriter() === append to a file
            fileWriter = new FileWriter(filename, true);
            BufferedWriter bufferWritter = new BufferedWriter(fileWriter);
            bufferWritter.write(result.toString());
            bufferWritter.close();

        }catch(IOException e){
            e.printStackTrace();
        }
    }//writeToCSV()
}//class

/*
    http://burlap.cs.brown.edu/doc/burlap/domain/singleagent/gridworld/GridWorldRewardFunction.html
    http://burlap.cs.brown.edu/doc/burlap/debugtools/MyTimer.html

*/
    /*

    //-----------------------------BEST SMALL World best settings--------------------------------------
    LearningRate learningRate = new ExponentialDecayLR(0.9, 0.99998, 0.0001);

		LearningAgent agent = new QLearning(domain, discountRate, hashingFactory, qInitial, learningRate_const);
        EpsilonGreedy epsilonObj = new EpsilonGreedy((QProvider) agent, epsilon);
        ((QLearning)agent).setLearningPolicy(epsilonObj);
        ((QLearning)agent).setLearningRateFunction(learningRate);

        Episode e = null;
		//run learning for 50 episodes
        MyTimer timer = new MyTimer();
		for(int i = 1; i <= 20000; i++){
            timer.start();
            e = agent.runLearningEpisode(env);
            timer.stop();
            iterTime = timer.getTime();

			if(i % 1000 == 0){
				epsilon -= 0.05;
				if(epsilon < 0) epsilon = 0.0;
				epsilonObj = new EpsilonGreedy((QProvider) agent, epsilon);
				((QLearning)agent).setLearningPolicy(epsilonObj);
			}

			//e.write(outputPath + "ql_" + i);
			//System.out.println(i+1 + ": " + e.maxTimeStep() + "\t" + iterTime + " s");

			//reset to an initial state from its current state, which may be a terminal state
			env.resetEnvironment();
		}


		// -----------------------------BEST LARGE WORLD----------------------------------//
		 LearningRate learningRate = new ExponentialDecayLR(0.9, 0.999999, 0.001);
        //learningRate = new SoftTimeInverseDecayLR(1.0, 5, 0.0001);

		LearningAgent agent = new QLearning(domain, discountRate, hashingFactory, qInitial, learningRate_const);
        EpsilonGreedy epsilonObj = new EpsilonGreedy((QProvider) agent, epsilon);
        ((QLearning)agent).setLearningPolicy(epsilonObj);
        ((QLearning)agent).setLearningRateFunction(learningRate);

        Episode e = null;
		//run learning for 50 episodes
        MyTimer timer = new MyTimer();
		for(int i = 1; i <= 100000; i++){
            timer.start();
            e = agent.runLearningEpisode(env);
            timer.stop();
            iterTime = timer.getTime();
            //numActions = e.numActions();

			if(i % 10000 == 0){
				epsilon -= 0.1;
				if(epsilon < 0) epsilon = 0.0;
				epsilonObj = new EpsilonGreedy((QProvider) agent, epsilon);
				((QLearning)agent).setLearningPolicy(epsilonObj);
			}

			//e.write(outputPath + "ql_" + i);
			//System.out.println(i+1 + ": " + e.maxTimeStep() + "\t" + iterTime + " s");

			//reset to an initial state from its current state, which may be a terminal state
			env.resetEnvironment();
		}

*/

/*
		learningRate = new ExponentialDecayLR(0.5, 0.999, 0.001);
        epsilonObj = new EpsilonGreedy((QProvider) agent, 0.00);
        ((QLearning)agent).setLearningPolicy(epsilonObj);
		((QLearning)agent).setLearningRateFunction(learningRate);
        for(int i = 0; i < 5000; i++){
            timer.start();
            e = agent.runLearningEpisode(env);
            timer.stop();
            iterTime = timer.getTime();

            //e.write(outputPath + "ql_" + i);
            //System.out.println(i+1 + ": " + e.maxTimeStep() + "\t" + iterTime + " s");

            //reset to an initial state from its current state, which may be a terminal state
            env.resetEnvironment();
        }
*/
