# Letter Recognition

[<img alt="FOURMOND Jérôme" src="https://avatars2.githubusercontent.com/u/15089371" width="100">](https://github.com/jfourmond) |
------------------------|
[@jfourmond](https://github.com/jfourmond	) |

This program use neural networks for Letter Recognition.

It uses the [DeepLearning4J](https://deeplearning4j.org/) library, and a transform result CSV from the dataset taken from [the UC Irvine Machine Learning Repository](http://archive.ics.uci.edu/ml/).

## Data Converter

The [DataConverter](https://github.com/jfourmond/LetterRecognition/blob/master/src/main/java/converter/DataConverter.java) class can be run in order to generate files which can be concatenate to make a [full CSV usable with the neural network](https://github.com/jfourmond/LetterRecognition/blob/master/src/main/resources/letter-recognition.csv).

## Letter Recognition Neural Network

The [LetterRecognitionNeuralNetwork](https://github.com/jfourmond/LetterRecognition/blob/master/src/main/java/converter/LetterRecognitionNeuralNetwork.java) class can be run in order to visualize and test your configuration :
- Number of layers
- Number of hidden neurons per layers
- Number of outputs neurons per layers
- Learning Rate
- Activation Function
- Number of iterations
- Seed
- ...

Once launch, you can observe the evolution of your neural network and this url : [http://localhost:9000](http://localhost:9000)

## Resources

In order to run the [DataConverter](https://github.com/jfourmond/LetterRecognition/blob/master/src/main/java/converter/DataConverter.java), you need to follow a few steps :
1. Download [winutils](https://github.com/steveloughran/winutils/blob/master/hadoop-2.7.1/bin/winutils.exe)
2. Save [winutils.exe](https://github.com/steveloughran/winutils/blob/master/hadoop-2.7.1/bin/winutils.exe) binary to a directory of your choice (e.g. c:\hadoop\bin).
3. Specify to the program one argument : the path where bin/winutils.exe can be found.
4. Run
5. If the program runned successfully : generated file can be found in a directory *"data_processed\_[timestamp]"* as *part-00000* and *part-00001*. You just have to concatenate the two files to get a full csv format file.
