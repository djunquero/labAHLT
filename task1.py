from xml.dom.minidom import parse
from nltk import tokenize
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
            # tokenlist = tokenize(text)
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


def tokenize(text):
    return


def evaluate(inputdir, outputfile):
    return


if __name__ == "__main__":
    nerc(INPUT_DIR, OUTPUT_FILE)
