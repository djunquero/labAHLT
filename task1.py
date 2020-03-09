from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
from rules import *
import os
import re
from tqdm import tqdm

import nltk
nltk.download('punkt')
INPUT_DIR = os.path.join(os.path.dirname(__file__), 'data\Train')
OUTPUT_FILE = "task9.1_output_3.txt"
DRUG_SUFFIXES = "suffixes_drug.txt"
DRUG_PREFIXES = "prefixes_drugs.txt"

def nerc(inputdir, outputfile):
    open(outputfile, "w").close()
    for file in tqdm(os.listdir(inputdir)):
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
            if token[0].endswith(suffix) and len(token[0]) > 4:
                entities.append(build_entity(token, "drug"))
                break
        # Rule 2: check drug prefixes
        for prefix in drug_prefixes:
            if token[0].startswith(prefix) and len(token[0]) > 4:
                entities.append(build_entity(token, "drug"))
                break
        # Rule 3: Check drug contained
        for contained in drug_contained:
            if contained in token[0]:
                entities.append(build_entity(token, "drug"))
                break
        # Rule 4: Check drug_n suffixes
        for suffix in drug_n_suffixes:
            if token[0].endswith(suffix) and len(token[0]) > 2:
                entities.append(build_entity(token, "drug_n"))
                break
        # Rule 5: Check drug_n contained
            for contained in drug_n_contained:
                if contained in token[0]:
                    entities.append(build_entity(token, "drug_n"))
                    break
        # Rule 6: Full capital letters without numbers and longer than 4 words, they are brands
        if token[0].isupper() and not re.search(r'\d', token[0]) and len(token[0]) > 4:
            entities.append(build_entity(token, "brand"))

        # Rule 7: Group suffixes
        for suffix in group_suffixes:
            if token[0].endswith(suffix) and len(token[0]) > 3:
                entities.append(build_entity(token, "group"))
                break
        # Rule 8: Group prefixes
            for prefix in group_prefixes:
                if token[0].startswith(prefix) and len(token[0]) > 3:
                    entities.append(build_entity(token, "group"))
                    break
        # Rule 9: Words that have numbers and letters with more than 3 letters are drugs
        if re.search(r'\d', token[0]) and len(token[0]) > 4 and re.match("^[A-Za-z]*$", token[0]):
            entities.append(build_entity(token, "drug"))
        # Rule 10: Words ending in -s (possible plurals) that are very long (+8 characters) are groups
        if token[0].endswith('s') and len(token[0]) > 8:
            entities.append(build_entity(token, "group"))
        # Rule 10: Digit followed by dash combinations are found in drug_n
        if re.search(r'\d-', token[0]):
            entities.append(build_entity(token, "drug_n"))

        with open(DRUG_SUFFIXES) as f:
            for suffix in f:
                if token[0].endswith(suffix.rstrip()):
                    entities.append(build_entity(token, "drug"))

        with open(DRUG_PREFIXES) as f:
            for prefix in f:
                if token[0].startswith(prefix.rstrip()):
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
