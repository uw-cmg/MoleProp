# check if file has experimental properties in first layer
# return into nested layer on success None on failure
def hasExperimentalProperties(data):
    for i in range(len(data['Record']['Section'])):
        if data['Record']['Section'][i]['TOCHeading'] == 'Chemical and Physical Properties':
            for j in range(len(data['Record']['Section'][i]['Section'])):
                if data['Record']['Section'][i]['Section'][j]['TOCHeading'] == 'Experimental Properties':
                    return data['Record']['Section'][i]['Section'][j]
    return None

# check if it has flashpoint
# store all records of flashpoint in a list
# return that list else -1 on failure
def getFlashpoint(data):
    for i in range(len(data['Section'])):
        if data['Section'][i]['TOCHeading'] == 'Flash Point':
            return data['Section'][i]['Information']
    return None

    return None
# get compound name
def getCompoundName(data):
    for i in range(len(data['Record']['Section'])):
        if data['Record']['Section'][i]['TOCHeading'] == 'Names and Identifiers':
            for j in range(len(data['Record']['Section'][i]['Section'])):
                if data['Record']['Section'][i]['Section'][j]['TOCHeading'] == 'Computed Descriptors':
                    return data['Record']['Section'][i]['Section'][j]['Section'][0]['Information'][0]['Value']['StringWithMarkup'][0]['String']
    return None

# get SMILES
def getSmiles(data):
    for i in range(len(data['Record']['Section'])):
        if data['Record']['Section'][i]['TOCHeading'] == 'Names and Identifiers':
            for j in range(len(data['Record']['Section'][i]['Section'])):
                if data['Record']['Section'][i]['Section'][j]['TOCHeading'] == 'Computed Descriptors':
                    for k in range(len(data['Record']['Section'][i]['Section'][j]['Section'])):
                        if data['Record']['Section'][i]['Section'][j]['Section'][k]['TOCHeading'] == 'Canonical SMILES':
                            return data['Record']['Section'][i]['Section'][j]['Section'][k]['Information'][0]['Value']['StringWithMarkup'][0]['String']
    return None

def strToNegInt(num):
    num = num.lstrip('-')
    return int(num) * -1

def findUnit(data):
    array = data[0]['Value']['StringWithMarkup'][0]['String'].split()
    for value in array:     
        if value == 'F':
            return 'F'
        if value == 'C':
            return 'C'
        if value == 'Â°C':
            return 'C'

def findTemp(data):
    array = data[0]['Value']['StringWithMarkup'][0]['String'].split()
    if array[0] == 'Not' or array[0] == 'Flammable':
        return None
    for value in array:
        #check if value has a decimal
        if value.find('.') != -1:
            index = value.find('.')
            value = value[0:index]
        if value[0] == '-':
            temp = strToNegInt(value)
            break
        if value.isdigit() == True:
            temp = value
            break
    print(data[0]['Value']['StringWithMarkup'][0]['String'])
    return int(temp)

def farnenheitToCelsius(temp):
    return (temp - 32) * (5 / 9)

# finds state of the compound i.e solvent liquid solid etc.
def getState(data):
    return 0 #TODO
