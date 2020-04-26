from xml.dom.minidom import parse
from nltk.parse.corenlp import CoreNLPDependencyParser
import os
from tqdm import tqdm


# cd stanford-corenlp-full-2018-10-05
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer

import nltk
nltk.download('punkt')
INPUT_DIR = os.path.join(os.path.dirname(__file__), 'data\\Devel')
OUTPUT_FILE = "task9.2_output_1.txt"


def ddi(inputdir, outputfile):
    # ddi function is implemented as specified on the lab statement
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
                entities[entity_id] = entity_offset

            analysis = analyze(sentence_text)

            pairs = sentence.getElementsByTagName("pair")
            for pair in pairs:
                id_entity_1 = pair.attributes["e1"].value
                id_entity_2 = pair.attributes["e2"].value
                (is_ddi, ddi_type) = check_interaction(analysis, entities, id_entity_1, id_entity_2)
                file = open(outputfile, "a")
                file.write("|".join([sentence_id, id_entity_1, id_entity_2, is_ddi, ddi_type])+'\n')

    evaluate(inputdir, outputfile)


def analyze(sentence_text):

    # Core NLP is used. It sometimes throws StopIteration exception, in which case the analysis continues with the next
    # sentence
    parser = CoreNLPDependencyParser(url="http://localhost:9000")
    try:
        dependency_graph, = parser.raw_parse(sentence_text)

        # For every word, the offset is added to the corresponding node
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
    except StopIteration:
        return None


def check_interaction(analysis, entities, id_entity_1, id_entity_2):

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
        return "0", "null"

    entity_1 = analysis.get_by_address(entity_1_address)
    entity_2 = analysis.get_by_address(entity_2_address)


    # Weights for every type. Different rules may have different weights for each type, as some rules might
    # indicate a strong correlation between the pair and a type,
    # whereas others can be more generic (increases all types)
    effect = 0
    mechanism = 0
    interaction = 0
    advise = 0


    # Rule 1: An entity has the key words in the "head"
    if analysis.get_by_address(entity_1["head"])["word"] in ["effects"] or \
            analysis.get_by_address(entity_2["head"])["word"] in ["effects"]:
        effect += 2

    # Rule 2: If an entity is a noun modifier of the word "levels"
    if analysis.get_by_address(entity_1["head"])["word"] in ["levels"] and entity_1["rel"] == "nmod" or \
            analysis.get_by_address(entity_2["head"])["word"] in ["levels"] and entity_2["rel"] == "nmod":
        mechanism += 2

    # Rule 3: Clue words in betweeen for "effect"
    for address in range(entity_1_address, entity_2_address):
        if analysis.get_by_address(address)["lemma"] in ["administer", "potentiate", "prevent"]:
            effect += 2

    # Rule 4: Clue words in betweeen for "mechanism"
    for address in range(entity_1_address, entity_2_address):
        if analysis.get_by_address(address)["lemma"] in ["reduce", "increase", "decrease"]:
            mechanism += 2

    # Rule 5: Clue words in betweeen begining of the sentence and an entity for "interaction"
    for address in range(1, entity_2_address):
        if analysis.get_by_address(address)["lemma"] in ["interact", "interaction"]:
            interaction += 2

    # Rule 6: Clue words in betweeen begining of the sentence and an entity for "interaction"
    for address in range(1, entity_2_address):
        if analysis.get_by_address(address)["lemma"] == "drug" and analysis.get_by_address(analysis.get_by_address(address)["head"])["lemma"] == "interaction":
            interaction += 2

    # Rule 7: Clue words in betweeen for "advise"
    for address in range(entity_1_address, entity_2_address):
        if analysis.get_by_address(address)["lemma"] in ["should", "require", "recommend", "is"]:
            advise += 1

    # Heuristic requirement: An interaction will be considered only if enough rules passed.
    if effect + mechanism + interaction + advise < 2:
        return "0", "null"

    # The chosen type corresponds to the one with the highest weight
    interaction_type = max(effect, mechanism, interaction, advise)

    if interaction_type == effect:
        return "1", "effect"
    elif interaction_type == mechanism:
        return "1", "mechanism"
    elif interaction_type == interaction:
        return "1", "int"
    else:
        return "1", "advise"


def evaluate(inputdir, outputfile):
    """ Starts the evaluation process, which computes the nerc score.

            :param inputdir: Relative path to the training folder
            :param outputfile: File containing the predictions to be evaluated
    """
    os.system("java -jar eval/evaluateDDI.jar " + inputdir + " " + outputfile)


if __name__ == "__main__":
    ddi(INPUT_DIR, OUTPUT_FILE)
