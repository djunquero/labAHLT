from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
import os

'''
import nltk
nltk.download('punkt')
'''
INPUT_DIR = os.path.join(os.path.dirname(__file__), 'data\Train')
OUTPUT_FILE = "task_1_output.txt"


def nerc(inputdir, outputfile):
    for file in os.listdir(inputdir):
        tree = parse(os.path.join(inputdir, file))
        sentences = tree.getElementsByTagName("sentence")
        for sentence in sentences:
            (sentence_id, text) = getSentenceInfo(sentence)
            tokenlist = tokenize(text)
            entities = extract_entities(tokenlist)
            output_entities(sentence_id, entities, outputfile)
            for entity in entities:
                print(str(entity))

'''
        for sentence in tree:
            (sentence_id, text) = getSentenceInfo(sentence)
            tokenlist = tokenize(text)
            entities = extractentities(tokenlist)
            outputentities(sentence_id, entities, outputfile)
        
    evaluate(inputdir, outputfile)

'''


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
    # Rule 1: words ending in "-ane" are drugs
    for token in tokenlist:
        if token[0].endswith("ane"):
            entities.append(build_entity(token, "drug"))
    return entities


def build_entity(token, type):
    entity = {
        "name": token[0],
        "offset": str(token[1]) + "-" + str(token[2]),
        "type": type
    }
    return entity


def output_entities(id, ents, outf):
    return


def evaluate(inputdir, outputfile):
    return


if __name__ == "__main__":
    nerc(INPUT_DIR, OUTPUT_FILE)
