from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
import os
from tqdm import tqdm
from itertools import tee
import pycrfsuite

import nltk
nltk.download('punkt')
INPUT_DIR = os.path.join(os.path.dirname(__file__), 'data\Train')
OUTPUT_FILE = "task9.2_output_3.txt"
MODEL = "task9.2_model.crfsuite"


# Main function
def nerc(inputdir, outputfile):
    open(outputfile, "w").close()
    # for file in os.listdir(inputdir):
    for file in tqdm(os.listdir(inputdir)):
        tree = parse(os.path.join(inputdir, file))
        sentences = tree.getElementsByTagName("sentence")
        for sentence in sentences:
            (sentence_id, text) = getSentenceInfo(sentence)
            tokenlist = tee(tokenize(text), 3)
            features = extract_features(tokenlist[0])
            golden_entities = extract_golden_entities(tree)
            output_features(sentence_id, tokenlist[1], golden_entities, features, outputfile)
            # learner(features, golden_entities)
            # predicted_classes = classifier(features)
            # output_entities(sentence_id, tokenlist[2], predicted_classes, outputfile)

    evaluate(inputdir, outputfile)

'''
NOTE: Generators are exhausted after iteration over them. The tokenized sentence is copied as many times as it is
iterated upon. Generators are used instead of lists because the "yield" return allows to iterate efficiently when
generating the offsets.
'''


# Extracts the sentence ID and the text body from the XML attributes.
def getSentenceInfo(sentence):
    """ Extracts the sentence ID and the text body from the XML attributes.

            :param sentence: A sentence in XML format. Must contain "id" and "text" attributes
            :returns: A paired tuple with (id, text)
    """
    return sentence.getAttribute("id"), sentence.getAttribute("text")


def tokenize(text):
    """ Splits text into tokens, returning a generator with all the words and special characters obtained

            :param text: A String containing the text to be tokenized
            :returns: Generator containing tokens extracted from the sentence
    """
    tokens = word_tokenize(text)
    offset = 0
    for token in tokens:
        offset = text.find(token, offset)
        yield token, offset, offset + len(token)-1
        offset += len(token)


def extract_golden_entities(sentence):
    golden_entities = []
    entities = sentence.getElementsByTagName("entity")
    for entity in entities:
        golden_entities.append((entity.getAttribute("text"),
                                entity.getAttribute("charOffset"),
                                entity.getAttribute("type"))
                               )
    return golden_entities


def extract_features(tokenlist):
    """ Extracts features from a tokenized sentence and puts them into a binary vector.

            :param tokenlist: A generator contained all tokens in a sentence
            :returns: A list of binary feature vectors
    """
    feature_vectors = []
    previous_word = word = capitalized = ""
    try:
        next_word = next(tokenlist)[0]
        while True:
            try:
                word = next_word
                if word.isupper():
                    capitalized = "upper"
                elif word.islower():
                    capitalized = "lower"
                elif word[0].isupper():
                    capitalized = "capitalized"
                elif word.isnumeric():
                    capitalized = "numeric"
                else:
                    capitalized = "mixed"
                next_word = next(tokenlist)[0]
                feature_vectors.append(build_feature(word, previous_word, next_word, word[-4:], word[:4], capitalized))
                previous_word = word
            except StopIteration:
                feature_vectors.append(build_feature(word, previous_word, "", word[-4:], word[:4], capitalized))
                break
    except StopIteration:
        pass
    return feature_vectors


def build_feature(word, previous_word, next_word, suffix, prefix, capitalized):
    """ Creates a binary vector with the given attributes

            :param word: Core word that defines the token
            :param previous_word: The previous word found in the same sentence
            :param next_word: The next word found in the same sentence
            :param suffix: The last 4 letters of the word
            :param prefix: The first 4 letters of the word
            :param capitalized: A string describing the capitalization of the word (upper, lower, mixed, etc.)
            :returns: A binary feature vector
    """
    feature = [
        "word="+word,
        "previous=" + previous_word,
        "next=" + next_word,
        "suffix=" + suffix,
        "prefix=" + prefix,
        "capitalized=" + capitalized
    ]

    return feature


def output_features(sentence_id, entities, golden_entities, features, outputfile):
    """ Outputs all feature vectors with its associated sentence_id and offset

            :param sentence_id: Identifier of the sentence
            :param entities: The tokenized sentence as a generator, containing the token and offset
            :param golden_entities:  A list of tuples containing words belonging to some class, with the format
            :param features: A list of feature vectors
            :param outputfile: The relative file path where the output should be stored
            :returns: A string with feature vectors in each line
    """
    # file = open(outputfile, "a")
    token_index = 0
    entity_index = 0
    output = ""
    for entity in entities:
        gold_class = "0"
        form = entity[0]
        offset_start = str(entity[1])
        offset_end = str(entity[2])
        # Check if word is part of a compound entity
        if entity_index < len(golden_entities) and offset_start == golden_entities[entity_index][1].split('-')[0]:
            gold_class = golden_entities[entity_index][2]
            form = golden_entities[entity_index][0]
            offset = golden_entities[entity_index][1].split('-')
            offset_start = offset[0]
            offset_end = offset[1]
            entity_index += 1

        output = sentence_id + '\t' + form + '\t' + offset_start + '\t' + offset_end + '\t' + gold_class
        for feature in features[token_index]:
            output += '\t' + feature
        output += '\n'
        token_index += 1
        print(output)
    return output


def learner(features, golden_entities):
    """ Instanciates a crfsuite learner and trains the model.
        While it doesnt reaturn anything, it saves the model into a file

            :param features: A list of feature vectors
            :param golden_entities: A list of tuples containing words belonging to some class, with the format
            (text, offset, class)
    """

    trainer = pycrfsuite.Trainer(verbose=False)
    labels = []
    for feature in tqdm(features):
        label = "0"
        for entity in golden_entities:
            word = feature[0].split("word=")[1]
            if word == entity[0]:
                label = entity[2]
                break
        labels.append(label)

    # TODO: Define X_Train (features) and Y_Train (labels)
    trainer.append(features, labels)

    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    trainer.train(MODEL)
    print(len(trainer.logparser.iterations), trainer.logparser.iterations[-1])


def classifier(features):
    """ Creates a list with predicted labels for each feature, using a previously instanciated learner.

            :param features: A list of feature vectors
            :returns: A list of predicted classes for each feature
    """
    predicted_classes = []

    tagger = pycrfsuite.Tagger()
    tagger.open(MODEL)

    # TODO: Placeholder, feature vector might need to be prepared
    for feature in features:
        predicted_classes.append(tagger.tag(feature))

    return predicted_classes


def output_entities(sentence_id, entities, classes, outputfile):
    """ Outputs all sentences in an appropiate format to be evaluated.
        Doesn't return anything but it creates a text file with a line for every drug prediction.

            :param sentence_id: Identifier of the sentence
            :param entities: The tokenized sentence as a generator, containing the token and offset
            :param classes: List of predicted classes for each token/feature
            :param outputfile: The relative file path where the output should be stored
    """
    file = open(outputfile, "a")
    i = 0
    for entity in entities:
        file.write(sentence_id + '|' + entity[1] + '|' + entity[2] + '|' + classes[i] + '\n')
        i += 1


def evaluate(inputdir, outputfile):
    """ Starts the evaluation process, which computes the nerc score.

            :param inputdir: Relative path to the training folder
            :param outputfile: File containing the predictions to be evaluated
    """
    os.system("java -jar eval/evaluateNER.jar "+ inputdir + " " + outputfile)


if __name__ == "__main__":
    nerc(INPUT_DIR, OUTPUT_FILE)
