import requests
from urllib.request import urlopen
import PyPDF2
import re
from rdkit import Chem

def removeJunk(compound):
    compound = compound.split(',')
    return compound[0]

def kelvinToCelsius(temp):
    return temp -  273.15

def celsiusToKelvin(temp):
    return temp + 273.15

def getSmiles(compound):
    smilesUrl = "https://opsin.ch.cam.ac.uk/opsin/"+compound+".smi"
    requestSmiles = requests.get(smilesUrl)
    if requestSmiles.status_code == 400 or requestSmiles.status_code == 404:
        return None
    else:
        return Chem.MolToSmiles(Chem.MolFromSmiles(requestSmiles.text))
    
def removeJunk(compound):
    compound = compound.split(',')
    return compound[0]
    
def getFlashPoint(pdfUrl):
    flashp = -1
    response = urlopen(pdfUrl)
    file = open("document.pdf", 'wb')
    file.write(response.read())
    file.close()
    pdf_file = open('document.pdf', 'rb')
        
    read_pdf = PyPDF2.PdfFileReader(pdf_file)
    
    #number_of_pages = read_pdf.getNumPages()
    for p in range(read_pdf.getNumPages()):
        page = read_pdf.getPage(p)
        text = page.extractText()
        
        text = text.replace('\n', ' ').replace('\r', '')
        fp = re.search('Flash Point(.+?)C', text)
        #print(fp)
        if fp != None:
            fp = re.search('Flash Point(.+?)C', text).group(0)
            flashPoint = re.findall('\d+', fp)
            #print(int(flashPoint[0]))
            flashp = int(flashPoint[0])
            break
    
    pdf_file.close()
    
    print(flashp)    
    return flashp

# returns product form, physical state, and chemical family
def compoundProperties(pdfUrl):
    family = None
    response = urlopen(pdfUrl)
    file = open("document.pdf", 'wb')
    file.write(response.read())
    file.close()
    pdf_file = open('document.pdf', 'rb')
        
    read_pdf = PyPDF2.PdfFileReader(pdf_file)
    response = urlopen(pdfUrl)
    page = read_pdf.getPage(0)
    text = page.extractText()
    text = text.replace('\n', ' ').replace('\r', '')

    fp = re.search('Product form(.+?) P', text)
    string = fp.group(0)
    strArray = string.split()
    form = strArray[3]
   #print(form)
    #fp = re.search('Physical state(.+?) F', text)
    #string = fp.group(0)
    #strArray = string.split()
    #state = strArray[3]
    #print(state)
    #fp = re.search('Chemical family(.+?) 1', text)
    #if fp != None:
    #    string = fp.group(0)
    #    strArray = string.split()
    #    family = strArray[3]
    #    print(family)
    if form == 'Substance':
        num = 1
    if form == 'Mixture':
        num = 0
    return form, num