import pandas as pd

def get_original_data(data_path):
    df = pd.read_csv(data_path, delimiter='\t', encoding="ISO-8859-1", names=['id1', 'id2', 'sentiment', 'tweet_content'])
    return df

def get_formatted_data(data_path):
    df = pd.read_csv(data_path, delimiter=',', header=0)
    return df

