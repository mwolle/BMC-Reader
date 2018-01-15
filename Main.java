package bmc_Reader.main;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.datexis.annotator.Annotator;
import de.datexis.annotator.AnnotatorFactory;
import de.datexis.common.ExportHandler;
import de.datexis.common.ObjectSerializer;
import de.datexis.common.Resource;
import de.datexis.encoder.Encoder;
import de.datexis.encoder.EncoderSet;
import de.datexis.encoder.impl.LetterNGramEncoder;
import de.datexis.encoder.impl.PositionEncoder;
import de.datexis.encoder.impl.SurfaceEncoder;
import de.datexis.model.Annotation;
import de.datexis.model.Dataset;
import de.datexis.model.Document;
import de.datexis.model.tag.BIOESTag;
import de.datexis.models.ner.MentionAnnotation;
import de.datexis.models.ner.MentionAnnotator;
import de.datexis.models.ner.eval.MentionAnnotatorEval;
import de.datexis.models.ner.tagger.MentionTagger;
import de.datexis.models.ner.tagger.MentionTaggerIterator;
import de.datexis.preprocess.DocumentFactory;

public class Main {

	protected final static Logger log = LoggerFactory.getLogger(Main.class);

	//please edit texoo.properties and add the path where your models are stored
	private final static Resource path = Resource.fromConfig("de.datexis.path.models");
	private static final char DEFAULT_SEPARATOR = ',';
	private static final char DEFAULT_QUOTE = '"';

	public static void main(String[] args) throws Exception {


		System.out.println(args[0]); // Path to documents ( parsed documents )
		System.out.println(args[1]); // Path to exactmatch ( exact match from hive )
		System.out.println(args[2]); // count of files/chapters to process
		
		int maxDocsCount = Integer.parseInt(args[2]);

		
		
		Annotator annotator;
		MentionTagger tagger;
	//	EncoderSet encoders;
		LetterNGramEncoder trigram = new LetterNGramEncoder(3);
		try {
			annotator = AnnotatorFactory.fromXML(path, "annotator.xml");
			
			tagger = (MentionTagger) annotator.getTagger();
			
		
		//	encoders = tagger.getEncoders();
		
		//	trigram.loadModel(path.resolve("trigram.tsv.gz"));
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			// generate neu model
		
			tagger = new MentionTagger("BLSTM"); // a mention tagger tags NER mentions :-)
			//   tagger.con
			//   tagger.loadModel(path.resolve("blstm.bin")); // LSTM was trained, needs to be saved

			annotator = new MentionAnnotator(tagger);
			//e.printStackTrace();			
			
		//	PositionEncoder position = new PositionEncoder(); // word position
		//	SurfaceEncoder surface = new SurfaceEncoder(); // word capitalization
		//	trigram = new LetterNGramEncoder(3); // letter-trigrams
		//	trigram.setName("trigram");
		//	encoders = new EncoderSet(position, surface, trigram); 
		//	for(Encoder enc : encoders) annotator.addComponent(enc);
		//	tagger.setEncoders(encoders);
		}

		
		int runBegun = 0;
		String exactMatchFile = args[1];

		String documentsFile = args[0];
		
		
		String currentDocument = "";
		String currentChapter = "";
		
		Document trainDoc = new Document(); // = DocumentFactory.fromText( "empty" );
		Document testDoc = trainDoc;
		int docCount = 0;
		Scanner scanner = new Scanner(new File(exactMatchFile));
		while (scanner.hasNext()) {
			List<String> line = parseLine(scanner.nextLine());
			System.out.println("DocID " 
									+ line.get(0) 
									+ ", code= " 
									+ line.get(1) 
									+ " , name=" 
									+ line.get(2) 
									+ "]"
									+ line.get(3) 
									+ line.get(4) 
									+ line.get(5) 
									+ line.get(6) 
									+ line.get(7) 
									+ line.get(8) 
									+ line.get(9) 
									+ line.get(10));
			if (!currentDocument.equals(line.get(0))|| !currentChapter.equals(line.get(2))) {
				docCount++;
				if (runBegun == 1) {
					// lernen

					// add Tags to Tokens based on Annotation
					MentionAnnotation.createTagsFromAnnotations(trainDoc, Annotation.Source.GOLD, BIOESTag.class);
					MentionAnnotation.createTagsFromAnnotations(testDoc, Annotation.Source.GOLD, BIOESTag.class);

					// add this Document into Datasets - for a real example we need around 4.000 training sentences
					Dataset train = new Dataset("training data", Arrays.asList(trainDoc));
					Dataset test  = new Dataset("test data", Arrays.asList(testDoc));
					// --- configuration -------------------------------------------------------
					// configure and train Encoders
					PositionEncoder position = new PositionEncoder(); // word position
					SurfaceEncoder surface = new SurfaceEncoder(); // word capitalization
					// already loaded // LetterNGramEncoder trigram = new LetterNGramEncoder(3); // letter-trigrams
					trigram.trainModel(train.getDocuments()); // collect all trigrams that occur in the training set
					EncoderSet encoders = new EncoderSet(position, surface, trigram); // put all of them together as feature vector

					
					// configure the recurrent neural network for training
					// already loaded // MentionTagger tagger = new MentionTagger("BLSTM"); // a mention tagger tags NER mentions :-)
					tagger.setTagset(BIOESTag.class); // we use BIOES or BIO2 labels for NER
					tagger.setEncoders(encoders); // set our features
					// This Annotator contains all the stuff from above and creates MentionAnnotations
					// already loaded //  MentionAnnotator annotator = new MentionAnnotator(tagger);
					for(Encoder enc : encoders) annotator.addComponent(enc);
			
					
					// --- training ------------------------------------------------------------
					// LSTM hyperparameters (good values for real world examples in brackets)
					int ffwLayers = 300;        // size of the feed-forward layers (300-500)
					int lstmLayers = 100;       // size of the LSTM layers (100-300)
					int examples = -1;         // number of sentences to use for training (-1 = all)
					int batchSize = 1;         // batch size in sentences (16-32)
					int iterations = 1;        // number of times every sentence is trained (1)
					int numEpochs = 10;       // number of times the whole dataset is trained (1-10)
					double learningRate = 0.01;  // learning rate (0.001-0.01)

					// create the Tagger iterator and finally train the model
					MentionTaggerIterator trainIt = new MentionTaggerIterator(
							train.getDocuments(), train.getName(), encoders, tagger.getTagset(), 
							Annotation.Source.GOLD, examples, batchSize, true
							);

					tagger = tagger.build(ffwLayers, lstmLayers, iterations, learningRate * batchSize);
					tagger.setTrainingParams(batchSize, numEpochs, true);
					tagger.trainModel(trainIt);
					//tagger.setListeners(enableUI()); // show training curve

					// --- test ----------------------------------------------------------------
					// test the model
					// AnnotationEval eval = new AnnotationEval(annotator.getTagger().getName(), train, test, Annotation.Source.GOLD, Annotation.Source.PRED);
					//MentionAnnotatorEval eval = new MentionAnnotatorEval(annotator.getTagger().getName(), train, test);
					MentionAnnotatorEval eval = new MentionAnnotatorEval(annotator.getTagger().getName());
					annotator.annotate(testDoc);
					eval.setTestDataset(test, 1, 1);
					eval.setTrainDataset(train, iterations, 1);
					eval.evaluateAnnotations();
					tagger.appendTestLog(eval.printAnnotationStats());
					// Retrieve all Annotations and print predictions
					for(MentionAnnotation ann : testDoc.getAnnotations(Annotation.Source.GOLD, MentionAnnotation.class)) {
						System.out.println(String.format("-- %s [%s]\t%s", ann.getText(), ann.getType(), ann.getConfidence()));
						System.out.println(ObjectSerializer.getJSON(ann));
					}
					if (docCount > maxDocsCount ) break;
					
					
				}
				try {
				trainDoc = DocumentFactory.fromText( getDocumentCorpus( documentsFile, line.get(0), line.get(2) ));
				} catch (Exception e) {
					runBegun = 0;
					break;
				}

				testDoc = trainDoc;
				
				currentDocument = line.get(0) ;
				currentChapter = line.get(2) ;
			}
			// single learning/annotationmode
			
			runBegun = 1;
			
		

			// --- preprocessing -------------------------------------------------------
			// parse texts into Documents
		//	String testWord = line.get(6);
		//	String tokenTest = trainDoc.getToken(Integer.parseInt(line.get(5))-1).toString();
			

			// add some GOLD annotations
			trainDoc.addAnnotation(new MentionAnnotation(Annotation.Source.GOLD, Arrays.asList(trainDoc.getToken(Integer.parseInt(line.get(5))-1).get()))); // TeXoo
			
		
		}
		scanner.close();
		
		// --- save ----------------------------------------------------------------
		// save models to a new path, you can configure this in texoo.properties
		Resource outputPath = ExportHandler.createExportPath("ExampleNER");
		System.out.println("saving model to path: " + outputPath);
		trigram.saveModel(outputPath, "trigram"); // trigrams were trained, need to be saved
		tagger.saveModel(outputPath, "blstm"); // LSTM was trained, needs to be saved
		annotator.writeModel(outputPath, "annotator"); // save the complete model as XML
		annotator.writeTrainLog(outputPath); // write log data
		annotator.writeTestLog(outputPath);

	}
	
	public static String getDocumentCorpus(String filePath, String documentName, String chapter ) throws Exception{
		// TODO needs to be optimized ... really
		String returnText = "";
		Scanner scanner = new Scanner(new File(filePath));
		while (scanner.hasNext()) {
			List<String> line = parseLine(scanner.nextLine());
			if (line.get(1).equals(documentName) && line.get(3).equals(chapter)) {
				returnText = line.get(6);
				break;
			}
				
		}
		scanner.close();
		if (returnText.equals("")) {
			throw new Exception();
		}
		return returnText;
	}


	public static List<String> parseLine(String cvsLine) {
		return parseLine(cvsLine, DEFAULT_SEPARATOR, DEFAULT_QUOTE);
	}

	public static List<String> parseLine(String cvsLine, char separators) {
		return parseLine(cvsLine, separators, DEFAULT_QUOTE);
	}

	public static List<String> parseLine(String cvsLine, char separators, char customQuote) {

		List<String> result = new ArrayList<>();

		//if empty, return!
		if (cvsLine == null && cvsLine.isEmpty()) {
			return result;
		}

		if (customQuote == ' ') {
			customQuote = DEFAULT_QUOTE;
		}

		if (separators == ' ') {
			separators = DEFAULT_SEPARATOR;
		}

		StringBuffer curVal = new StringBuffer();
		boolean inQuotes = false;
		boolean startCollectChar = false;
		boolean doubleQuotesInColumn = false;

		char[] chars = cvsLine.toCharArray();

		for (char ch : chars) {

			if (inQuotes) {
				startCollectChar = true;
				if (ch == customQuote) {
					inQuotes = false;
					doubleQuotesInColumn = false;
				} else {

					//Fixed : allow "" in custom quote enclosed
					if (ch == '\"') {
						if (!doubleQuotesInColumn) {
							curVal.append(ch);
							doubleQuotesInColumn = true;
						}
					} else {
						curVal.append(ch);
					}

				}
			} else {
				if (ch == customQuote) {

					inQuotes = true;

					//Fixed : allow "" in empty quote enclosed
					if (chars[0] != '"' && customQuote == '\"') {
						//curVal.append('"');
					}

					//double quotes in column will hit this!
					if (startCollectChar) {
						//curVal.append('"');
					}

				} else if (ch == separators) {

					result.add(curVal.toString());

					curVal = new StringBuffer();
					startCollectChar = false;

				} else if (ch == '\r') {
					//ignore LF characters
					continue;
				} else if (ch == '\n') {
					//the end, break!
					break;
				} else {
					curVal.append(ch);
				}
			}

		}

		result.add(curVal.toString());

		return result;
	}

}
