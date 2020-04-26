from xml.dom.minidom import parse
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.classify import MaxentClassifier
import os
from tqdm import tqdm


# cd stanford-corenlp-full-2018-10-05
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer

import nltk
nltk.download('punkt')
TRAIN_INPUT_DIR = os.path.join(os.path.dirname(__file__), 'data\\Train')
DEVEL_INPUT_DIR = os.path.join(os.path.dirname(__file__), 'data\\Devel')
OUTPUT_FILE = "task9.2_output_2.txt"


def ddi(train_inputdir, devel_inputdir, outputfile):

    training_vector = feature_extractor(train_inputdir, True)
    test_vector = feature_extractor(devel_inputdir, False)

    featuresets = []
    for featureset in training_vector:
        featuresets = featuresets + featureset[1]
    classifier = MaxentClassifier.train(featuresets, algorithm="iis", max_iter=100)

    file = open(outputfile, "w")
    for featureset in test_vector:
        result = classifier.classify(featureset[1])
        if result == "null":
            file.write(featureset[0] + '|0|' + result + '\n')
        else:
            file.write(featureset[0] + '|1|' + result + '\n')

    evaluate(DEVEL_INPUT_DIR, outputfile)


def feature_extractor(inputdir, training=True):
    vector = []
    for file in tqdm(os.listdir(inputdir)):
        tree = parse(os.path.join(inputdir, file))
        sentences = tree.getElementsByTagName("sentence")
        for sentence in sentences:
            sentence_id = sentence.getAttribute("id")
            sentence_text = sentence.getAttribute("text")

            entities = {}
            ents = sentence.getElementsByTagName("entity")

            for entity in ents:
                entity_id = entity.attributes["id"].value
                entity_offset = entity.attributes["charOffset"].value.split("-")
                entities[entity_id] = entity_offset

            try:
                analysis = analyze(sentence_text)

                pairs = sentence.getElementsByTagName("pair")
                for pair in pairs:
                    id_entity_1 = pair.attributes["e1"].value
                    id_entity_2 = pair.attributes["e2"].value
                    features = extract_features(analysis, entities, id_entity_1, id_entity_2)

                    if pair.hasAttribute("type"):
                        interaction_type = pair.attributes["type"].value
                    else:
                        interaction_type = "null"
                    vector.append(output_features(sentence_id, id_entity_1, id_entity_2,
                                                  interaction_type, features, training))
            except AttributeError:
                pass
            except StopIteration:
                pass
    return vector


def analyze(sentence_text):
    parser = CoreNLPDependencyParser(url="http://localhost:9000")
    dependency_graph, = parser.raw_parse(sentence_text)

    address = 0
    offset = 0
    while dependency_graph.contains_address(address):
        node = dependency_graph.get_by_address(address)
        word = node["word"]
        if isinstance(word, str):
            offset = sentence_text.find(word, offset)
            node["start"] = offset
            node["end"] = offset + len(word) - 1
            offset += len(word)
        address += 1

    return dependency_graph


def extract_features(analysis, entities, id_entity_1, id_entity_2):

    features = []

    address = 0
    entity_1_address = None
    entity_2_address = None

    # Finding nodes for each entity
    while analysis.contains_address(address):
        node = analysis.get_by_address(address)
        if isinstance(node["word"], str):
            if entities[id_entity_1][0] == str(node["start"]):
                entity_1_address = address
            elif entities[id_entity_2][0] == str(node["start"]):
                entity_2_address = address
            if entity_1_address is not None and entity_2_address is not None:
                break
        address += 1

    if entity_1_address is None or entity_2_address is None:
        raise AttributeError

    entity_1 = analysis.get_by_address(entity_1_address)
    entity_2 = analysis.get_by_address(entity_2_address)

    if entity_1_address > 1:
        features.append("lb1=" + analysis.get_by_address(entity_1_address-1)["lemma"])
    if entity_1_address > 2:
        features.append("lb1=" + analysis.get_by_address(entity_1_address-2)["lemma"])

    features.append("la1=" + analysis.get_by_address(entity_1_address+1)["lemma"])
    features.append("lb2=" + analysis.get_by_address(entity_2_address-1)["lemma"])

    la2 = analysis.get_by_address(entity_2_address+1)["lemma"]
    if la2 is not None:
        features.append("la2=" + la2)
    la2 = analysis.get_by_address(entity_2_address+2)["lemma"]
    if la2 is not None:
        features.append("la2=" + la2)

    features.append("h1=" + str(analysis.get_by_address(entity_1["head"])["lemma"]))
    features.append("h2=" + str(analysis.get_by_address(entity_2["head"])["lemma"]))
    features.append("h1r=" + str(analysis.get_by_address(entity_1["head"])["rel"]))
    features.append("h2r=" + str(analysis.get_by_address(entity_2["head"])["rel"]))

    return features


def output_features(sentence_id, id_entity_1, id_entity_2, interaction_type, features, training):
    if training:
        featureset = [(word_feats(features), interaction_type)]
    else:
        featureset = word_feats(features)
    # print(sentence_id + ' ' + id_entity_1 + ' ' + id_entity_2 + ' ' + interaction_type + ' ' + str_features[:-1])
    vector = ["|".join([sentence_id, id_entity_1, id_entity_2]), featureset]
    return vector


def word_feats(features):
    return dict([(feature, True) for feature in features])


def evaluate(inputdir, outputfile):
    """ Starts the evaluation process, which computes the nerc score.

            :param inputdir: Relative path to the training folder
            :param outputfile: File containing the predictions to be evaluated
    """
    os.system("java -jar eval/evaluateDDI.jar " + inputdir + " " + outputfile)


if __name__ == "__main__":
    ddi(TRAIN_INPUT_DIR, DEVEL_INPUT_DIR, OUTPUT_FILE)
