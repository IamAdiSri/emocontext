#Please use python 3.5 or above
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, LSTM, Bidirectional
from keras.layers.merge import concatenate
from keras import optimizers
from keras.models import load_model, Model
import json, argparse, os
import re
import io
import sys

# custom imports
from pickler import Pickler

# Path to training and testing data file. This data can be downloaded from a link, details of which will be provided.
trainDataPath = ""
testDataPath = ""
# Output file that will be generated. This file can be directly submitted.
solutionPath = ""
# Path to directory where GloVe file is saved.
gloveDir = ""
# Path to directory where SSE file is saved.
sseDir = ""
# Path to directory holding the word lists
wordlistsDir = ""

NUM_FOLDS = None                   # Value of K in K-fold Cross Validation
NUM_CLASSES = None                 # Number of classes - Happy, Sad, Angry, Others
MAX_NB_WORDS = None                # To set the upper limit on the number of tokens extracted using keras.preprocessing.text.Tokenizer 
MAX_SEQUENCE_LENGTH = None         # All sentences having lesser number of words than this will be padded
GLOVE_DIM = None                   # The dimension of the GloVe embeddings
FEATURE_DIM = None                 # Number of manual features for every word
SSE_DIM = None                     # The dimension of the Sentiment-Specific embeddings
BATCH_SIZE = None                  # The batch size to be chosen for training the model.
LSTM_DIM = None                    # The dimension of the representations learnt by the LSTM model
DROPOUT = None                     # Fraction of the units to drop for the linear transformation of the inputs. Ref - https://keras.io/layers/recurrent/
NUM_EPOCHS = None                  # Number of epochs to train a model for


label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}


def preprocessData(dataFilePath, mode):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    indices = []
    conversations = []
    labels = []
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            repeatedChars = ['.', '?', '!', ',']
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '    
                line = cSpace.join(lineSplit)
            
            line = line.strip().split('\t')
            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[line[4]]
                labels.append(label)
            
            conv = ' <eos> '.join(line[1:4])
            
            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)
            
            indices.append(int(line[0]))
            conversations.append(conv.lower())
    
    if mode == "train":
        return indices, conversations, labels
    else:
        return indices, conversations


def getMetrics(predictions, ground):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification  
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions.argmax(axis=1))
    
    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)
    
    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)
    
    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, NUM_CLASSES):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))
    
    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))   
    
    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()    
    
    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))
    
    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)
    
    microF1 = ( 2 * microRecall * microPrecision ) / (microPrecision + microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------
    
    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions==ground)
    
    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1


def writeNormalisedData(dataFilePath, texts):
    """Write normalised data to a file
    Input:
        dataFilePath : Path to original train/test file that has been processed
        texts : List containing the normalised 3 turn conversations, separated by the <eos> tag.
    """
    normalisedDataFilePath = dataFilePath.replace(".txt", "_normalised.txt")
    with io.open(normalisedDataFilePath, 'w', encoding='utf8') as fout:
        with io.open(dataFilePath, encoding='utf8') as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                line = line.strip().split('\t')
                normalisedLine = texts[lineNum].strip().split('<eos>')
                fout.write(line[0] + '\t')
                # Write the original turn, followed by the normalised version of the same turn
                fout.write(line[1] + '\t' + normalisedLine[0] + '\t')
                fout.write(line[2] + '\t' + normalisedLine[1] + '\t')
                fout.write(line[3] + '\t' + normalisedLine[2] + '\t')
                try:
                    # If label information available (train time)
                    fout.write(line[4] + '\n')    
                except:
                    # If label information not available (test time)
                    fout.write('\n')


class Wordlists():
    def __init__(self):
        self.features = FEATURE_DIM

        self.hedges       =  sorted( Pickler.load( os.path.join(wordlistsDir, 'hedges.pkl') ) )
        self.factives     =  sorted( Pickler.load( os.path.join(wordlistsDir, 'factives.pkl') ) )
        self.assertives   =  sorted( Pickler.load( os.path.join(wordlistsDir, 'assertives.pkl') ) )
        self.implicatives =  sorted( Pickler.load( os.path.join(wordlistsDir, 'implicatives.pkl') ) )
        self.reports      =  sorted( Pickler.load( os.path.join(wordlistsDir, 'reports.pkl') ) )
        self.entailments  =  sorted( Pickler.load( os.path.join(wordlistsDir, 'entailments.pkl') ) )
        self.subjectives  =  Pickler.load( os.path.join(wordlistsDir, 'subjectives.pkl') )
        self.polarities   =  Pickler.load( os.path.join(wordlistsDir, 'polarity.pkl') )

    def getWordfeatures(self, word):
        """Extract all the features for the input word and return the feature vector
        Input:
            word : word string of the word for which features are to be extracted
        Output:
            fv : feature vector for the input word
        """
        fv = [0]*self.features

        if word in self.hedges:
            fv[0] = 1
        if word in self.factives:
            fv[1] = 1
        if word in self.assertives:
            fv[2] = 1
        if word in self.implicatives:
            fv[3] = 1
        if word in self.reports:
            fv[4] = 1
        if word in self.entailments:
            fv[5] = 1
        try:
            subj = self.subjectives[word]
            if subj['pol'] == 'positive':
                fv[6] = 1
            if subj['type'] == 'strongsubj':
                fv[7] = 1
        except:
            pass
        try:
            if self.polarities[word] == 'positive':
                fv[8] = 1
        except:
            pass
        
        return fv
            
def getFSEM(wordIndex):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        fseMatrix : A matrix where every row has 309 dimensional GloVe + features embedding
    """
    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    with io.open(os.path.join(gloveDir, 'glove.6B.300d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector
    
    print('Found %s word vectors.' % len(embeddingsIndex))
    
    wordLists = Wordlists()
    # Minimum word index of any word is 1. 
    fseMatrix = np.zeros((len(wordIndex) + 1, GLOVE_DIM + FEATURE_DIM))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        featureVector = wordLists.getWordfeatures(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros for the embedding vector, although some features if found will be recorded.
            fseMatrix[i] = np.concatenate((embeddingVector, featureVector))
        else:
            fseMatrix[i] = np.concatenate(([0]*GLOVE_DIM, featureVector))
    
    return fseMatrix

def getSSEM(wordIndex):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        sseMatrix : A matrix where every row has 50 dimensional sentiment specific embedding
    """
    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    with io.open(os.path.join(sseDir, 'sswe-u.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector
    
    print('Found %s word vectors.' % len(embeddingsIndex))
    
    wordLists = Wordlists()
    # Minimum word index of any word is 1. 
    sseMatrix = np.zeros((len(wordIndex) + 1, SSE_DIM))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros for the embedding vector, although some features if found will be recorded.
            sseMatrix[i] = embeddingVector
    
    return sseMatrix


def buildModel(fseMatrix, sseMatrix):
    """Constructs the architecture of the model
    Features, BiLSTM
    Input:
        fseMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    input1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    input2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    fseLayer = Embedding(fseMatrix.shape[0],
                                GLOVE_DIM + FEATURE_DIM,
                                weights=[fseMatrix],
                                input_length=MAX_SEQUENCE_LENGTH)
    fseLayer.trainable=False
    fseLayer = fseLayer(input1)
    
    sseLayer = Embedding(sseMatrix.shape[0],
                                SSE_DIM,
                                weights=[sseMatrix],
                                input_length=MAX_SEQUENCE_LENGTH)
    sseLayer.trainable=False
    sseLayer = sseLayer(input2)

    hidden1 = LSTM(LSTM_DIM, dropout=0.5, return_sequences=True)(fseLayer)
    hidden2 = LSTM(LSTM_DIM, dropout=0.5, return_sequences=True)(sseLayer)
    # print("hidden shapes", hidden1._keras_shape, hidden2._keras_shape)

    merge = concatenate([hidden1, hidden2])
    # print("merge shapes", merge._keras_shape)

    intermediary = LSTM(LSTM_DIM, dropout=0.5)(merge)
    intermediary = LSTM(LSTM_DIM, dropout=0.5)(intermediary)
    # bilstm = Bidirectional(intermediary)
    # bilstm = Bidirectional(LSTM(LSTM_DIM, dropout=0.5))(merge)

    # dl = Dense(16, activation='relu')(bilstm)
    # dl = Dense(16, activation='relu')(dl)

    # output = Dense(NUM_CLASSES, activation='sigmoid')(dl)
    output = Dense(NUM_CLASSES, activation='sigmoid')(intermediary)

    # model = Sequential()
    # model.add(embeddingLayer)
    # model.add(Bidirectional(LSTM(LSTM_DIM, dropout=DROPOUT)))
    # model.add(Dense(NUM_CLASSES, activation='sigmoid'))
    
    model = Model(inputs=[input1, input2], outputs=[output])

    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])

    # plot_model(model, to_file='architecture.png')
    
    return model    

def main():
    parser = argparse.ArgumentParser(description="Baseline Script for SemEval")
    parser.add_argument('-config', help='Config to read details', required=True)
    args = parser.parse_args()

    with open(args.config) as configfile:
        config = json.load(configfile)
        
    global trainDataPath, testDataPath, solutionPath, gloveDir, sseDir, wordlistsDir
    global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, GLOVE_DIM, FEATURE_DIM, SSE_DIM
    global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE    
    
    trainDataPath = config["train_data_path"]
    testDataPath = config["test_data_path"]
    solutionPath = config["solution_path"]
    gloveDir = config["glove_dir"]
    sseDir = config["sse_dir"]
    wordlistsDir = config["wordlists_dir"]
    
    NUM_FOLDS = config["num_folds"]
    NUM_CLASSES = config["num_classes"]
    MAX_NB_WORDS = config["max_nb_words"]
    MAX_SEQUENCE_LENGTH = config["max_sequence_length"]
    GLOVE_DIM = config["embedding_dim"]
    FEATURE_DIM = config["feature_dim"]
    SSE_DIM = config["sse_dim"]
    BATCH_SIZE = config["batch_size"]
    LSTM_DIM = config["lstm_dim"]
    DROPOUT = config["dropout"]
    LEARNING_RATE = config["learning_rate"]
    NUM_EPOCHS = config["num_epochs"]
        
    print("Processing training data...")
    trainIndices, trainTexts, labels = preprocessData(trainDataPath, mode="train")
    # Write normalised text to file to check if normalisation works. Disabled now. Uncomment following line to enable   
    # writeNormalisedData(trainDataPath, trainTexts)
    print("Processing test data...")
    testIndices, testTexts = preprocessData(testDataPath, mode="test")
    # writeNormalisedData(testDataPath, testTexts)

    print("Extracting tokens...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(trainTexts)
    trainSequences = tokenizer.texts_to_sequences(trainTexts)
    testSequences = tokenizer.texts_to_sequences(testTexts)

    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))

    print("Populating feature augmented semantic embedding matrix...")
    fseMatrix = getFSEM(wordIndex)

    print("Populating sentiment specific embedding matrix...")
    sseMatrix = getSSEM(wordIndex)

    data = pad_sequences(trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))
    print("Shape of training data tensor: ", data.shape)
    print("Shape of label tensor: ", labels.shape)
        
    # Randomize data
    np.random.shuffle(trainIndices)
    data = data[trainIndices]
    labels = labels[trainIndices]
      
    # Perform k-fold cross validation
    metrics = {"accuracy" : [],
               "microPrecision" : [],
               "microRecall" : [],
               "microF1" : []}
    
    # print("Starting k-fold cross validation...")
    # for k in range(NUM_FOLDS):
    #     print('-'*40)
    #     print("Fold %d/%d" % (k+1, NUM_FOLDS))
    #     validationSize = int(len(data)/NUM_FOLDS)
    #     index1 = validationSize * k
    #     index2 = validationSize * (k+1)
            
    #     xTrain = np.vstack((data[:index1],data[index2:]))
    #     yTrain = np.vstack((labels[:index1],labels[index2:]))
    #     xVal = data[index1:index2]
    #     yVal = labels[index1:index2]
    #     print("Building model...")
    #     model = buildModel(fseMatrix, sseMatrix)
    #     model.fit([xTrain, xTrain], yTrain, 
    #               validation_data=([xVal, xVal], yVal),
    #               epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    #     predictions = model.predict([xVal, xVal], batch_size=BATCH_SIZE)
    #     accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions, yVal)
    #     metrics["accuracy"].append(accuracy)
    #     metrics["microPrecision"].append(microPrecision)
    #     metrics["microRecall"].append(microRecall)
    #     metrics["microF1"].append(microF1)
        
    # print("\n============= Metrics =================")
    # print("Average Cross-Validation Accuracy : %.4f" % (sum(metrics["accuracy"])/len(metrics["accuracy"])))
    # print("Average Cross-Validation Micro Precision : %.4f" % (sum(metrics["microPrecision"])/len(metrics["microPrecision"])))
    # print("Average Cross-Validation Micro Recall : %.4f" % (sum(metrics["microRecall"])/len(metrics["microRecall"])))
    # print("Average Cross-Validation Micro F1 : %.4f" % (sum(metrics["microF1"])/len(metrics["microF1"])))
    
    # print("\n======================================")
    
    print("Retraining model on entire data to create solution file")
    model = buildModel(fseMatrix, sseMatrix)
    model.fit([data, data], labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    model.save('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))
    # model = load_model('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))

    print("Creating solution file...")
    testData = pad_sequences(testSequences, maxlen=MAX_SEQUENCE_LENGTH)
    predictions = model.predict([testData, testData], batch_size=BATCH_SIZE)
    predictions = predictions.argmax(axis=1)

    with io.open(solutionPath, "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')        
        with io.open(testDataPath, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(label2emotion[predictions[lineNum]] + '\n')
    print("Completed. Model parameters: ")
    print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d" 
          % (LEARNING_RATE, LSTM_DIM, DROPOUT, BATCH_SIZE))

               
if __name__ == '__main__':
    main()
