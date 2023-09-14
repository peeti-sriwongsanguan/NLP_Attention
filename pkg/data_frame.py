
import pandas as pd
import re
pd.set_option('max_columns', None)



def eda_nltk(dataframe):
    
    if dataframe.split('.')[-1] == 'json' or dataframe.split('.')[-1] == 'csv' or dataframe.split('.')[-1] == 'xlsx':

        if dataframe.split('.')[-1] == 'json':
            df = pd.read_json(dataframe, lines=True)
#             print(df.head())
#             print('*'*60)
        elif dataframe.split('.')[-1] == 'csv':
            df = pd.read_csv(dataframe)
#             print(df.head())
#             print('*'*60)
        elif dataframe.split('.')[-1] == 'xlsx':
            df = pd.read_excel(dataframe)
#             print(df.head())
#             print('*'*60)
        else:
            print('Check your file name')
        return df
    
    else:
       
        if dataframe.split('.')[-2]  == 'json':
            df = pd.read_json(dataframe, lines=True)
#             print(df.head())
#             print('*'*60)
        elif dataframe.split('.')[-2]  == 'csv':
            df = pd.read_csv(dataframe)
#             print(df.head())
#             print('*'*60)
        elif dataframe.split('.')[-2]  == 'xlsx':
            df = pd.read_excel(dataframe)
#             print(df.head())
#             print('*'*60)
        else:
            print('Check your file name')
    
        return df
    
    
def remove_tags(string):
    removelist = ""
    result = re.sub('','',string)          #remove HTML tags
    result = re.sub('https://.*','',result)   #remove URLs
    result = re.sub(r'\W+', ' ',result)    #remove non-alphanumeric characters 
    result = result.lower()
    return result
    
