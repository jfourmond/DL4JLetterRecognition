package converter;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Date;
import java.util.List;

/**
 * Execute this class with program arguments for convert the "letter-recognition.data.txt" file
 * into a csv file usable by the LetterRecognitionNeuralNetwork
 * 1 argument needed :
 * - Hadoop directory : the path where "bin/winutils.exe" can be found
 */
public class DataConverter {
    private static Logger Log = LoggerFactory.getLogger(DataConverter.class);

    private static String hadoopPath;   // "C:\\Program Files (x86)\\Hadoop"

    private static final String FILENAME = "letter-recognition.data.txt";

    private static File file;
    private static String baseDir;
    private static String inputPath;
    private static String outputPath;

    public static void main(String args[]) {
        if (args.length == 1)
            hadoopPath = args[0];
        else
            throw new IllegalArgumentException("At least one argument is necessary : the path where bin/winutils.exe" +
                    "can be found.");

        System.setProperty("hadoop.home.dir", hadoopPath);

        try {
            file = new ClassPathResource(FILENAME).getFile();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        baseDir = file.getParent();
        inputPath = file.getAbsolutePath();
        String timeStamp = String.valueOf(new Date().getTime());
        outputPath = baseDir + "\\data_processed_" + timeStamp;

        /**
         *  1.	lettr	capital letter	(26 values from A to Z)
         *  2.	x-box	horizontal position of box	(integer)
         *  3.	y-box	vertical position of box	(integer)
         *  4.	width	width of box			(integer)
         *  5.	high 	height of box			(integer)
         *  6.	onpix	total # on pixels		(integer)
         *  7.	x-bar	mean x of on pixels in box	(integer)
         *  8.	y-bar	mean y of on pixels in box	(integer)
         *  9.	x2bar	mean x variance			(integer)
         *  10.	y2bar	mean y variance			(integer)
         *  11.	xybar	mean x y correlation		(integer)
         *  12.	x2ybr	mean of x * x * y		(integer)
         *  13.	xy2br	mean of x * y * y		(integer)
         *  14.	x-ege	mean edge count left to right	(integer)
         *  15.	xegvy	correlation of x-ege with y	(integer)
         *  16.	y-ege	mean edge count bottom to top	(integer)
         *  17.	yegvx	correlation of y-ege with x	(integer)
         */

        /* Declaring Schema */
        Schema inputDataSchema = new Schema.Builder()
                .addColumnCategorical("lettr", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
                        "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")
                .addColumnsInteger("x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar",
                        "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-edge", "yegvx")
                .build();

        /* Declaring transformation process */
        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .categoricalToInteger("lettr")  /* Convert the categorical columns "lettr" into an Integer */
                .build();

        /* Declaring Spark Configuration */
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("Letter Recognition Data Reader Transform");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        JavaRDD<String> lines = sc.textFile(inputPath);
        JavaRDD<List<Writable>> letterData = lines.map(new StringToWritablesFunction(new CSVRecordReader()));
        JavaRDD<List<Writable>> processed = SparkTransformExecutor.execute(letterData, tp);
        JavaRDD<String> toSave = processed.map(new WritablesToStringFunction(","));

        toSave.saveAsTextFile(outputPath);

        sc.close();
    }

}
