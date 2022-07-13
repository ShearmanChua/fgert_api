import requests
import json
from typing import List, Dict
import pandas as pd
import ast

def predict_fget(dataset: Dict):
    response = requests.post('http://0.0.0.0:8000/df_link', json = dataset)
    
    #ipdb.set_trace()
    # print([doc for doc in response.iter_lines()])
    response = response.json()
    # print(type(response))
    # df_json = json.loads(response)
    # print(type(df_json))
    df = pd.json_normalize(response, max_level=0)

    print(df.head())
    print(df.info())

    df.to_csv("data/test_fget.csv",index=False)

    return df


if __name__ == '__main__':

    df = pd.read_csv("data/test.csv")

    print(df.head())
    print(df.info())

    df_json = df.to_json(orient="records")
    df_json = json.loads(df_json)
    jerex_results = predict_fget(df_json)
    jerex_results.to_csv("fgET_results.csv", index=False)
    