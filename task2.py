from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
import os
from tqdm import tqdm

import nltk
nltk.download('punkt')
INPUT_DIR = os.path.join(os.path.dirname(__file__), 'data\Train')
OUTPUT_FILE = "task9.1_output_3.txt"

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
