import time
import json
import codecs
import os
from os import listdir
from os.path import isfile, join
from tempfile import gettempdir
from argparse import ArgumentParser
from io import StringIO
from collections import defaultdict
import random
from operator import sub
from telnetlib import Telnet
import ast
import requests


from pandas import array
import pandas as pd
import numpy as np
import tqdm

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import torch
torch.multiprocessing.set_start_method('spawn')
from torch.utils.data import DataLoader

from model.fgET_model import fgET
from model.fgET_data import FetDataset
from model.fgET_preprocessor import PreProcessor
import util

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

from fastapi import FastAPI, Request

app = FastAPI()

initialize(config_path="configs", job_name="app")
cfg = compose(config_name="configs")
print(OmegaConf.to_yaml(cfg))

util.config_to_abs_paths(cfg.file_paths, 'labels_file_name')
util.config_to_abs_paths(cfg.file_paths, 'elmo_option_file')
util.config_to_abs_paths(cfg.file_paths, 'elmo_weights_file')
util.config_to_abs_paths(cfg.file_paths, 'test_file_name')
util.config_to_abs_paths(cfg.file_paths, 'model_checkpoint_file_name')
util.config_to_abs_paths(cfg.file_paths, 'save_path')

@app.get("/")
async def root():
    return {"message": "fgET API"}

def inference(cfg) -> None:

    num_worker = 1

    labels_file_path = cfg.file_paths.labels_file_name
    elmo_option = cfg.file_paths.elmo_option_file
    elmo_weight = cfg.file_paths.elmo_weights_file

    # load Fine-grained Entity Typing Labels
    with open(labels_file_path) as json_file:
        labels_strtoidx = json.load(json_file)

    labels_idxtostr = {i: s for s, i in labels_strtoidx.items()}
    label_size = len(labels_strtoidx)
    print('Label size: {}'.format(len(labels_strtoidx)))

    if cfg.test:
        print("------------ Performing model testing!!!! ------------")
        test_file_path = cfg.file_paths.test_file_name
        labels_strtoidx['No label'] = label_size + 1
        labels_idxtostr[label_size + 1] = 'No label'
        preprocessor = PreProcessor(labels_strtoidx,
                                    elmo_option=elmo_option,
                                    elmo_weight=elmo_weight,
                                    elmo_dropout=cfg.elmo_dropout)

        test_set = FetDataset(preprocessor,test_file_path,cfg.tokens_field,cfg.entities_field,cfg.sentence_field,labels_strtoidx,cfg.gpu)
        test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,collate_fn=preprocessor.batch_process,num_workers=num_worker)

        # Set GPU device
        gpu = torch.cuda.is_available() and cfg.gpu
        if gpu:
            torch.cuda.set_device(cfg.device)

        # Build model
        model = fgET(label_size,
                    elmo_dim = preprocessor.elmo_dim,
                    repr_dropout=cfg.repr_dropout,
                    dist_dropout=cfg.dist_dropout,
                    latent_size=cfg.latent_size,
                    svd=cfg.svd
                    )
        if gpu:
            model.cuda()

        total_step = len(test_loader)

        optimizer = model.configure_optimizers(cfg.weight_decay,cfg.lr,total_step)

        model_file_path = cfg.file_paths.model_checkpoint_file_name
        checkpoint = torch.load(model_file_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for name, param in model.named_parameters():print(name, param)

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': vars(cfg),
            'vocab': {'label': labels_strtoidx}
        }

        results = run_test(test_loader,model,cfg.gpu)

        collated_results = {'results':[]}
        for gold, pred, men_id,mention,sentence,score in zip(results['gold'],results['pred'],results['ids'],results['mentions'],results['sentence'],results['scores']):
            arranged_results = dict()

            pred_labels = [labels_idxtostr[i] for i, l in enumerate(pred) if l]

            arranged_results['mention_id'] = men_id
            arranged_results['mention'] = mention
            arranged_results['sentence'] = sentence
            arranged_results['predictions'] = pred_labels
            arranged_results['scores'] = [score[i]for i, l in enumerate(pred) if l]

            collated_results['results'].append(arranged_results)

        training_data = collated_results['results']
        training_records = {}

        for i in range(0,len(training_data)):
            training_records[str(i)] = training_data[i]

        json_object = json.dumps(training_records, indent = 4)
        df = pd.read_json(StringIO(json_object), orient ='index')
        
        return df
        

def run_test(test_loader,model,gpu=False):

    progress = tqdm.tqdm(total=len(test_loader), mininterval=1,
                        desc='Test')

    results = defaultdict(list)
    with torch.no_grad():
        for batch in test_loader:

            elmo_embeddings, labels, men_masks, ctx_masks, dists, gathers, men_ids, mentions,sentences = batch

            if gpu:
                elmo_embeddings = elmo_embeddings.to(device='cuda')
                labels = torch.cuda.FloatTensor(labels)
                men_masks = torch.cuda.FloatTensor(men_masks)
                ctx_masks = torch.cuda.FloatTensor(ctx_masks)
                gathers = torch.cuda.LongTensor(gathers)
                dists = torch.cuda.FloatTensor(dists)

            else:
                labels = torch.FloatTensor(labels)
                men_masks = torch.FloatFloatTensorTensor(men_masks)
                ctx_masks = torch.LongTensor(ctx_masks)
                gathers = torch.LongTensor(gathers)
                dists = torch.FloatTensor(dists)


            progress.update(1)

            preds,scores = model.predict(elmo_embeddings, men_masks, ctx_masks, dists, gathers)
            results['gold'].extend(labels.int().data.tolist())
            results['pred'].extend(preds.int().data.tolist())
            results['scores'].extend(scores.tolist())
            results['ids'].extend(men_ids)
            results['mentions'].extend(mentions)
            results['sentence'].extend(sentences)

    progress.close()

    return results

@app.post("/df_link")
async def link(request: Request):
    df_dict_str = await request.json()
    # df_json = json.dumps(df_dict_str)
    # df = pd.read_json(df_json, orient="records")
    df = pd.json_normalize(df_dict_str, max_level=0)
    print(df.head())
    print(df.info())
    
    df.to_csv(os.path.join(cfg.file_paths.save_path,"temp.csv"),index=False)

    results_df = inference(cfg)
    print(results_df.head())

    try:
        os.remove(os.path.join(cfg.file_paths.save_path,"temp.csv"))
    except:
        print("cannot remove temp")

    df_json = results_df.to_json(orient="records")
    df_json = json.loads(df_json)

    return df_json
