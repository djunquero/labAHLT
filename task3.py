from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
from rules import *
import os
import re
from tqdm import tqdm

import nltk
nltk.download('punkt')
INPUT_DIR = os.path.join(os.path.dirname(__file__), 'data\Train')
OUTPUT_FILE = "task9.2_output_1.txt"


def nerc(inputdir, outputfile):
    open(outputfile, "w").close()
    for file in tqdm(os.listdir(inputdir)):
        tree = parse(os.path.join(inputdir, file))
        sentences = tree.getElementsByTagName("sentence")
        for sentence in sentences:
            sentence_id = sentence.getAttribute("id").value
            sentence_text = sentence.getAttribute("text").value

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
                (is_ddi, ddi_type) = check_interaction(analysis, entities, id_entity_1, id_entity_2)
                print("|".join([sentence_id, id_entity_1, id_entity_2, is_ddi, ddi_type]), file=outputfile)

    evaluate(inputdir, outputfile)


def analyze(sentence_text):
    tokens = word_tokenize(sentence_text)
    offset = 0
    for token in tokens:
        offset = text.find(token, offset)
        yield token, offset, offset + len(token)-1
        offset += len(token)

    return []


def check_interaction(analysis, entities, id_entity_1, id_entity_2):
    return []


def evaluate(inputdir, outputfile):
    """ Starts the evaluation process, which computes the nerc score.

            :param inputdir: Relative path to the training folder
            :param outputfile: File containing the predictions to be evaluated
    """
    os.system("java -jar eval/evaluateNER.jar "+ inputdir + " " + outputfile)


if __name__ == "__main__":
    nerc(INPUT_DIR, OUTPUT_FILE)
