from xml.dom.minidom import parse, parseString

def nerc(inputdir, outputfile) :
    for file in inputdir :
        tree = parseXML(file)
        for sentence in tree :
            (id, text) = getsentenceinfo(sentence)
            tokenlist = tokenize(text)
            entities = extractentities(tokenlist)
            outputentities(id, entities, outputfile)
        
    evaluate(inputdir,outputfile)