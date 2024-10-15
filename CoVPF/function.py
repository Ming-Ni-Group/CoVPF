from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_F
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam, Adam
import pandas as pd
import datetime
import warnings
from collections import Counter, OrderedDict, defaultdict
from timeit import default_timer
from typing import List
from pyro import poutine
from pyro.poutine.util import site_is_subsample
from pyro.infer.reparam import LocScaleReparam
from pyro.nn.module import PyroModule, PyroParam

def process_fitting(BasicModel, counts, features, affinity, escape, device, place_lineage_index,time_step_days):
    pyro_model= BasicModel(counts, features, affinity, escape, device, place_lineage_index,time_step_days)
    pyro_model = pyro_model.to(device)
    pyro_model.fit(lr=0.01, num_steps = 8000, log_every=100)

    result = torch.load('./result.pkl')
    alpha_PL = result['mean']['init'].data
    beta_PL = result['mean']['rate'].data
    time = (torch.arange(float(counts.shape[0])) * time_step_days).to(device)
    logits = alpha_PL + beta_PL * time[:, None, None]
    logits = torch_F.softmax(logits, dim = -1).data

    return logits


def process_forecast(BasicModel, counts, features, affinity, escape, device, place_lineage_index,time_step_days,period = 10):
    pyro_model= BasicModel(counts, features, affinity, escape, device, place_lineage_index).to(device)
    pyro_model.fit(lr=0.01, num_steps = 8000, log_every=100)

    result = torch.load('./result.pkl')
    alpha_PL = result['mean']['init'].data
    beta_PL = result['mean']['rate'].data
    time = (torch.arange(float(counts.shape[0]) + period) * time_step_days).to(device)
    logits = alpha_PL + beta_PL * time[:, None, None]
    logits = torch_F.softmax(logits, dim = -1).data   
    forecast = logits[-1*period:, :, :]#.unsqueeze(0)

    return forecast


def get_coefficient(BasicModel, counts, features, affinity, escape, device, place_lineage_index,time_step_days):
    pyro_model= BasicModel(counts, features, affinity, escape, device, place_lineage_index).to(device)
    pyro_model.fit(lr=0.01, num_steps = 8000, log_every=100)
    result = torch.load('./result.pkl')
    dic = {
    'rate' : result['mean']['rate'].data.cpu(),
    'init' : result['mean']['init'].data.cpu(),
    'coef' : result['mean']['coef'].data.cpu(),
    'rate_std' : result['std']['rate'].data.cpu(),
    'init_std' : result['std']['init'].data.cpu(),
    'coef_std' : result['std']['coef'].data.cpu(),
    }
    torch.save(dic, './coefficient_dataset_Site.pkl')
