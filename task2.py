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
            features = extract_features(tokenlist)
            for feature in features:
                print(str(feature))
            # output_entities(sentence_id, entities, outputfile)

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


def extract_features(tokenlist):
    feature_vectors = []
    previous_word = word = capitalized = ""
    iterator = iter(tokenlist)
    next_word = next(iterator)[0]
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
            next_word = next(iterator)[0]
            feature_vectors.append(build_feature(word, previous_word, next_word, word[-4:], word[:4], capitalized))
            previous_word = word
        except StopIteration:
            feature_vectors.append(build_feature(word, previous_word, "", word[-4:], word[:4], capitalized))
            break
    return feature_vectors


def build_feature(word, previous_word, next_word, suffix, prefix, capitalized):
    feature = [
        "word="+word,
        "previous=" + previous_word,
        "next=" + next_word,
        "suffix=" + suffix,
        "prefix=" + prefix,
        "capitalized=" + capitalized
    ]

    return feature


def output_entities(sentence_id, entities, outputfile):
    file = open(outputfile, "a")
    for entity in entities:
        file.write(sentence_id + '|' + entity["offset"] + '|' + entity["name"] + '|' + entity["type"] + '\n')


def evaluate(inputdir, outputfile):
    os.system("java -jar eval/evaluateNER.jar "+ inputdir + " " + outputfile)


if __name__ == "__main__":
    nerc(INPUT_DIR, OUTPUT_FILE)
