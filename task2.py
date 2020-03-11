from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
import os
from tqdm import tqdm
from itertools import tee

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
            tokenlist = tee(tokenize(text), 2)
            features = extract_features(tokenlist[0])
            output_features(sentence_id, tokenlist[1], features, outputfile)

    evaluate(inputdir, outputfile)


def getSentenceInfo(sentence):
    return sentence.getAttribute("id"), sentence.getAttribute("text")


def tokenize(text):
    tokens = word_tokenize(text)
    offset = 0
    for token in tokens:
        offset = text.find(token, offset)
        yield token, offset, offset + len(token)-1
        offset += len(token)


def extract_features(tokenlist):
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
    feature = [
        "word="+word,
        "previous=" + previous_word,
        "next=" + next_word,
        "suffix=" + suffix,
        "prefix=" + prefix,
        "capitalized=" + capitalized
    ]

    return feature


def output_features(sentence_id, entities, features, outputfile):
    # file = open(outputfile, "a")
    i = 0
    for entity in entities:
        output = sentence_id + '\t' + entity[0] + '\t' + str(entity[1]) + '\t' + str(entity[2])
        for feature in features[i]:
            output += '\t' + feature
        output += '\n'
        i += 1
        print(output)
        # file.write(sentence_id + '|' + entity["offset"] + '|' + entity["name"] + '|' + entity["type"] + '\n')


def evaluate(inputdir, outputfile):
    os.system("java -jar eval/evaluateNER.jar "+ inputdir + " " + outputfile)


if __name__ == "__main__":
    nerc(INPUT_DIR, OUTPUT_FILE)
