from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
from rules import drugs_suffixes, drug_n_suffixes
import os
import re

import nltk
nltk.download('punkt')
INPUT_DIR = os.path.join(os.path.dirname(__file__), 'data\Train')
OUTPUT_FILE = "task9.1_output_3.txt"


def nerc(inputdir, outputfile):
    open(outputfile, "w").close()
    for file in os.listdir(inputdir):
        tree = parse(os.path.join(inputdir, file))
        sentences = tree.getElementsByTagName("sentence")
        for sentence in sentences:
            (sentence_id, text) = getSentenceInfo(sentence)
            tokenlist = tokenize(text)
            entities = extract_entities(tokenlist)
            output_entities(sentence_id, entities, outputfile)

    evaluate(inputdir, outputfile)


def getSentenceInfo(sentence):
    return sentence.getAttribute("id"), sentence.getAttribute("text")


def tokenize(txt):
    tokens = word_tokenize(txt)
    offset = 0
    for token in tokens:
        offset = txt.find(token, offset)
        yield token, offset, offset + len(token)
        offset += len(token)


def extract_entities(tokenlist):
    entities = []
    for token in tokenlist:
        # Rule 1: check drug suffixes
        for suffix in drugs_suffixes:
            if token[0].endswith(suffix) and len(token[0]) > 3 and len(token[0]) > 3:
                entities.append(build_entity(token, "drug"))
                break
        # Rule 2: Check drug_n suffixes
        for suffix in drug_n_suffixes:
            if token[0].endswith(suffix) and len(token[0]) > 3:
                entities.append(build_entity(token, "drug_n"))
                break
        # Rule 3: Full capital letters without numbers are brands
        if token[0].isupper() and not re.search(r'\d', token[0]) and len(token[0]) > 3:
            entities.append(build_entity(token, "brand"))
        #Rule 4: Groups
        # Rule 5: Words that have numbers and letters with more than 3 letters are drugs
        if re.search(r'\d', token[0]) and len(token[0]) >= 3 and re.match("^[A-Za-z]*$", token[0]):
            entities.append(build_entity(token, "drug"))
    return entities


def build_entity(token, type):
    entity = {
        "name": token[0],
        "offset": str(token[1]) + "-" + str(token[2]-1),
        "type": type
    }
    return entity


def output_entities(sentence_id, entities, outputfile):
    file = open(outputfile, "a")
    for entity in entities:
        file.write(sentence_id + '|' + entity["offset"] + '|' + entity["name"] + '|' + entity["type"] + '\n')


def evaluate(inputdir, outputfile):
    os.system("java -jar eval/evaluateNER.jar "+ inputdir + " " + outputfile)


if __name__ == "__main__":
    nerc(INPUT_DIR, OUTPUT_FILE)
