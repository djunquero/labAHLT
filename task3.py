from xml.dom.minidom import parse
from nltk.parse.corenlp import CoreNLPDependencyParser
import os
from tqdm import tqdm


# cd stanford-corenlp-full-2018-10-05
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer

import nltk
nltk.download('punkt')
INPUT_DIR = os.path.join(os.path.dirname(__file__), 'data\Train')
OUTPUT_FILE = "task9.2_output_1.txt"


def ddi(inputdir, outputfile):
    open(outputfile, "w").close()
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
                entities[id] = entity_offset

            analysis = analyze(sentence_text)

            pairs = sentence.getElementsByTagName("pair")
            for pair in pairs:
                id_entity_1 = pair.attributes["e1"].value
                id_entity_2 = pair.attributes["e2"].value
                # (is_ddi, ddi_type) = check_interaction(analysis, entities, id_entity_1, id_entity_2)
                # print("|".join([sentence_id, id_entity_1, id_entity_2, is_ddi, ddi_type]), file=outputfile)

    evaluate(inputdir, outputfile)


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


def check_interaction(analysis, entities, id_entity_1, id_entity_2):
    return []


def evaluate(inputdir, outputfile):
    """ Starts the evaluation process, which computes the nerc score.

            :param inputdir: Relative path to the training folder
            :param outputfile: File containing the predictions to be evaluated
    """
    os.system("java -jar eval/evaluateDDI.jar "+ inputdir + " " + outputfile)


if __name__ == "__main__":
    ddi(INPUT_DIR, OUTPUT_FILE)
