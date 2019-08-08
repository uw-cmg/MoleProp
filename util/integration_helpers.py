from rdkit import Chem
import pandas as pd

def remove_duplicates_by_source(dataframe):
    sources = dataframe.source.unique()
    for source in sources:
        sourceframe = dataframe[dataframe['source'] == source]
        dataframe = dataframe[dataframe['source'] != source]
        sourceframe.drop_duplicates(subset ='smiles',keep='first', inplace=True)
        frames = [dataframe, sourceframe]
        dataframe = pd.concat(frames)
    return dataframe

def check_null_values(dataframe):
    return dataframe.isnull().sum()

def check_similarity(dataset1, dataset2):
    count = 0
    print('Number of data in ' + dataset1.iloc[2]['source'] + ': ' +  str(len(dataset1)))
    print('Number of data in ' + dataset2.iloc[2]['source'] + ': '+ str(len(dataset2)))
    for i in range(len(dataset1)):
        for j in range(len(dataset2)):
            if dataset1.iloc[i]['compound'] == dataset2.iloc[j]['compound']:
                count += 1
    sim1 = count/len(dataset1)
    sim2 = count/len(dataset2)
    print('similarity for '+ dataset1.iloc[2]['source'] + '= ' + str(sim1) +
          '\nsimilarity for ' + dataset2.iloc[2]['source'] + '= ' + str(sim2))
    return (sim1,sim2)

def remove_invalid_smiles(data):
    invalid = []
    for index, row in data.iterrows():
        #print(row['smiles'])
        if Chem.MolFromSmiles(row['smiles']) == None:
            invalid.append(row['smiles'])
            #data.drop(row.index[0], inplace=True)
    print('invalid smiles strings')
    print('------------------------------')
    print(invalid)
    print('------------------------------')
    for smi in invalid:
        data.drop([data[data['smiles'] == smi].index[0]], inplace=True)
    return data

def remove_duplicates(data):
    result = data.drop_duplicates(subset='smiles', keep=False)#[~duplicates]
    #for each unique smiles that has duplicates
    for smiles in data[data.duplicated(subset='smiles')]['smiles'].unique():
        dup_rows = data.loc[data['smiles'] == smiles]
        if dup_rows['flashpoint'].unique().shape[0] == 1:
            # remove all but one
            result = result.append(dup_rows.iloc[0], sort=False)
        else:
            if dup_rows['flashpoint'].std() < 5:
                # add 1 back
                result = result.append(dup_rows.iloc[0], sort=False)
    return result  

def canonicalize_smiles(data):
    for idx, row in data.iterrows():
        m = Chem.MolFromSmiles(data.iloc[idx]['smiles'])
        if m != None:
            data.iloc[idx]['smiles'] = Chem.MolToSmiles(m)
        else:
            data.iloc[idx]['smiles'] = None
    return data

def kelvinToCelsius(temp):
    return temp -  273.15

def celsiusToKelvin(temp):
    return temp + 273.15

def get_compound_with_element(data, element):
    return data[data['smiles'].str.contains(element)]
