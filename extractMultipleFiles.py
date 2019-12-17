from xml.dom import minidom
import pathlib
from xml.etree.ElementTree import Element
from xml.parsers.expat import ExpatError

f = open("datasheet.txt", "w+", encoding='utf-8')

for x in range(11):
    fileNum = x+1
    packageName = "Packet" + str(fileNum) + "Concensus.txt"
    file = open(packageName)

    for line in file:
        #print(file)
        values = line.split()
        output = ""
        if values[1] == "N":
            output = "0 "
        else:
            output = "1 "
        filename = values[0] + ".xml"
        file = pathlib.Path(filename)
        if file.exists():
            try:
                xmldoc = minidom.parse(filename)
            except:
                pass
            phrases = xmldoc.getElementsByTagName('body')
            for phrase in phrases:
                if phrase.firstChild is not None:
                    if phrase.firstChild.nodeType != 1:
                        #print(phrase.firstChild.nodeType)
                        #print(phrase.firstChild)
                        output += phrase.firstChild.wholeText
            #print(line + output)
            toPrint = True
            lst = list(f.readlines())
            if len(lst) != 0:
                last_line = lst[len(lst)-1]
                if last_line == output:
                    toPrint = False
            if (output != "") and toPrint:
                f.write(output)
                f.write("\n")


