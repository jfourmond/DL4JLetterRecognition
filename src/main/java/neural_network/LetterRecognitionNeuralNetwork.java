package neural_network;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class LetterRecognitionNeuralNetwork {
    private static Logger Log = LoggerFactory.getLogger(LetterRecognitionNeuralNetwork.class);

    private static final String FILENAME = "letter-recognition.csv";

    private static UIServer uiServer;
    private static StatsStorage statsStorage;

    private static DataSet allData;
    private static DataSet trainingData;
    private static DataSet testData;

    private static DataNormalization normalizer;

    private static final int numClasses = 26;
    private static final int numInputs = 16;
    private static final int numHidden = 52;
    private static final int numOutputs = 26;
    private static final double learningRate = 0.6;
    private static final int iterations = 100;
    private static final long seed = 13;

    private static MultiLayerNetwork net;

    private static void loadData() throws IOException, InterruptedException {
        int numLinesToSkip = 0;
        String delimiter = ",";

        int labelIndex = 0;
        int batchSize = 20000;

        Log.info("Loading dataset");

        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource(FILENAME).getFile()));

        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
        allData = iterator.next();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.75);
        trainingData = testAndTrain.getTrain();
        testData = testAndTrain.getTest();
    }

    private static void normalizeData() {
        Log.info("Normalizing data");

        normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);
        normalizer.transform(trainingData);
        normalizer.transform(testData);
    }

    private static void buildNetwork() {
        Log.info("Building model...");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation(Activation.SIGMOID)
                .weightInit(WeightInit.XAVIER)
                .learningRate(learningRate)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHidden)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numHidden).nOut(numHidden)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHidden).nOut(numOutputs).build())
                .backprop(true).pretrain(false)
                .build();

        Log.info("Building neural network...");
        net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(100), new StatsListener(statsStorage));
    }

    private static void train() {
        Log.info("Training neural network...");
        net.fit(trainingData);
    }

    private static void evaluate() {
        Log.info("Evaluate neural network...");

        Evaluation eval = new Evaluation(numClasses);
        INDArray output = net.output(testData.getFeatureMatrix());
        eval.eval(testData.getLabels(), output);

        Log.info(eval.stats());
    }

    public static void main(String args[]) throws IOException, InterruptedException {
        Log.info("Launching program.");

        uiServer = UIServer.getInstance();
        statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);

        loadData();
        normalizeData();
        buildNetwork();
        train();
        evaluate();

        Log.info("End of the program.");
    }

}
