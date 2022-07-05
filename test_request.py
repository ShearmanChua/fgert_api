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


def generate_entity_linking_df(results_df):

    entities_linking_df = pd.DataFrame(columns=['doc_id','mention', 'mention_type','context_left','context_right'])

    for idx, row in results_df.iterrows():
        doc_id = row['doc_id']
        if type(row['relations']) == str:
            relations = ast.literal_eval(row['relations'])
            tokens = ast.literal_eval(row['tokens'])
        else:
            relations =  row['relations']
            tokens = row['tokens']
        entities = []
        for relation in relations:
            head_entity = " ".join(tokens[relation['head_span'][0]:relation['head_span'][1]])
            if head_entity not in entities:
                print("Head Entity:")
                print(head_entity)
                left_context = " ".join(tokens[relation['head_span'][0]-100:relation['head_span'][0]])
                print("Left context: ",left_context)
                print("\n")
                right_context = " ".join(tokens[relation['head_span'][1]:relation['head_span'][1]+100])
                print("Right context: ",right_context)
                print("\n")
                entities_linking_df.loc[-1] = [doc_id, head_entity, relation['head_type'], left_context, right_context]  # adding a row
                entities_linking_df.index = entities_linking_df.index + 1  # shifting index
                entities_linking_df = entities_linking_df.sort_index()  # sorting by index
                entities.append(head_entity)

            tail_entity = " ".join(tokens[relation['tail_span'][0]:relation['tail_span'][1]])
            if tail_entity not in entities:
                print("Tail Entity:")
                print(tail_entity)
                left_context = " ".join(tokens[relation['tail_span'][0]-100:relation['tail_span'][0]])
                print("Left context: ",left_context)
                print("\n")
                right_context = " ".join(tokens[relation['tail_span'][1]:relation['tail_span'][1]+100])
                print("Right context: ",right_context)
                print("\n")
                entities_linking_df.loc[-1] = [doc_id, tail_entity, relation['tail_type'], left_context, right_context]  # adding a row
                entities_linking_df.index = entities_linking_df.index + 1  # shifting index
                entities_linking_df = entities_linking_df.sort_index()  # sorting by index
                entities.append(tail_entity)

    print(entities_linking_df.head())
    return entities_linking_df


if __name__ == '__main__':

    df = pd.read_csv("data/test.csv")

    print(df.head())
    print(df.info())

    df_json = df.to_json(orient="records")
    df_json = json.loads(df_json)
    jerex_results = predict_fget(df_json)
    jerex_results.to_csv("fgET_results.csv", index=False)
    