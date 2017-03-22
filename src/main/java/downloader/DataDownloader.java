package downloader;

import converter.DataConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileOutputStream;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;

/**
 * Execute this class to download directly in a specified file
 * 1 argument needed :
 * - Path to the output file (for example : C:\Users\letter-recognition.data")
 */
public class DataDownloader {
    private static Logger Log = LoggerFactory.getLogger(DataDownloader.class);

    private static final String URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data";

    private static String output;

    public static void main(String args[]) throws IOException {
        if (args.length == 1)
            output = args[0];
        else
            throw new IllegalArgumentException("At least one argument is necessary : the path to the output file");

        Log.info("Downloading \"letter recognition.data\" file from the UCI Machine Learning Repository");
        URL website = new URL(URL);
        ReadableByteChannel rbc = Channels.newChannel(website.openStream());
        FileOutputStream fos = new FileOutputStream(output);
        fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
    }
}
