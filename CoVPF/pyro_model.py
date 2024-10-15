import warnings
warnings.filterwarnings('ignore')
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
from collections import Counter, OrderedDict, defaultdict
from timeit import default_timer
from typing import List
from pyro import poutine
from pyro.poutine.util import site_is_subsample
from pyro.infer.reparam import LocScaleReparam
from pyro.nn.module import PyroModule, PyroParam
from train_data import total_dataset, train_dataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = total_dataset
### dataset = train_dataset

counts = dataset['counts']
features = dataset['mutation_features']
affinity = dataset['affinity_features']
escape = dataset['escape_features']
time_step_days = dataset["time_step_days"]

locations = dataset['locations']
lineages = dataset['lineages']
mutations = dataset['mutations']
place_lineage_index = dataset['place_lineage_index']

counts = counts.to(device)
features = features.to(device)
affinity = affinity.to(device)
escape = escape.to(device)
place_lineage_index = place_lineage_index.to(device)

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=100, bias=True)
        self.fc2 = nn.Linear(in_features=100, out_features=1, bias=True)
        self.relu = nn.ReLU()
    def forward(self, input):
        y = self.relu(self.fc1(input))
        y = self.fc2(y).squeeze(-1)
        return y
net_affinity = MLP(affinity.shape[2]).to(device)
net_escape = MLP(escape.shape[2]).to(device)

class BasicModel(nn.Module):
    def __init__(
        self,
        counts,
        features,
        affinity,
        escape,
        device,
        place_lineage_index,
        time_step_days
    ):
        super().__init__()
        self.device = device
        self.counts = counts
        self.features = features
        self.affinity = affinity
        self.escape = escape
        self.place_lineage_index = place_lineage_index
        
        T, P, L = counts.shape
        L, F = features.shape
        L, A, S= affinity.shape
        L, E, S = escape.shape
        PL = len(place_lineage_index)
        
        self.num_time_steps = T
        self.num_places = P
        self.num_lineages = L
        self.num_mutations = F
        self.num_affinity_sections = A
        self.num_escape_sections = E
        self.num_sites = S
        self.num_place_lineage_index = PL
        
    
    def model(self):
        pyro.module("net_affinity", net_affinity)
        pyro.module("net_escape", net_escape)

        T = self.num_time_steps
        P = self.num_places
        L = self.num_lineages
        F = self.num_mutations
        A = self.num_affinity_sections
        E = self.num_escape_sections
        S = self.num_sites
        PL = self.num_place_lineage_index

        counts = self.counts
        features = self.features
        affinity = self.affinity
        escape = self.escape
        place_lineage_index = self.place_lineage_index
        device = self.device

        time_plate = pyro.plate("time", T, dim=-3)
        place_plate = pyro.plate("place", P, dim=-2)
        lineage_plate = pyro.plate("lineage", L, dim=-1)
        pl_plate = pyro.plate("place_lineage", PL, dim=-1)

        # Configure reparametrization (which does not affect model density).
        reparam = {}
        reparam["coef"] = LocScaleReparam()
        reparam["beta_A"] = LocScaleReparam()
        reparam["beta_E"] = LocScaleReparam()
        reparam["pl_rate"] = LocScaleReparam()
        reparam["pl_init"] = LocScaleReparam()
        with poutine.reparam(config=reparam):
            # Sample global random variables.
            coef_scale = pyro.sample("coef_scale", dist.LogNormal(-4, 2))
            rate_scale = pyro.sample("rate_scale", dist.LogNormal(-4, 2))
            init_scale = pyro.sample("init_scale", dist.LogNormal(0, 2))
            affinity_scale = pyro.sample("affinity_scale", dist.LogNormal(-4, 2))
            escape_scale = pyro.sample("escape_scale", dist.LogNormal(-4, 2))

            coef = pyro.sample(
                "coef", dist.Laplace(torch.zeros(F), coef_scale).to_event(1)
            )  # [F]
            beta_A = pyro.sample(
                "beta_A", dist.Laplace(torch.zeros(A), affinity_scale).to_event(1)
            )
            beta_E = pyro.sample(
                "beta_E", dist.Laplace(torch.zeros(E), escape_scale).to_event(1)
            )

            coef_scale, rate_scale, init_scale, affinity_scale, escape_scale = coef_scale.to(device), rate_scale.to(device), init_scale.to(device), affinity_scale.to(device), escape_scale.to(device)
            coef, beta_A, beta_E = coef.to(device), beta_A.to(device), beta_E.to(device)

            with lineage_plate:
                affinity_input = net_affinity(affinity)
                escape_input = net_escape(escape)
                rate_loc = pyro.deterministic("rate_loc", coef @ features.T + 0.01 * beta_A @ affinity_input.T + 0.01 * beta_E @ escape_input.T)
            with pl_plate:
                pl_rate_loc = rate_loc.expand(P, L).reshape(-1)
                pl_rate = pyro.sample(
                    "pl_rate", dist.Normal(pl_rate_loc[place_lineage_index], rate_scale)
                )
                pl_init = pyro.sample("pl_init", dist.Normal(0, init_scale))
            with place_plate, lineage_plate:
                rate = pyro.deterministic(
                    "rate",
                    pl_rate_loc.scatter(0, place_lineage_index, pl_rate).reshape(P, L),
                ) 
                init = pyro.deterministic(
                    "init",
                    torch.full((P * L,), -1e2).to(device).scatter(0, place_lineage_index, pl_init).reshape(P, L),
                )
            rate_loc, pl_rate_loc, pl_rate, pl_init, rate, init = rate_loc.to(device), pl_rate_loc.to(device), pl_rate.to(device), pl_init.to(device), rate.to(device), init.to(device)
            time = torch.arange(float(T)) * time_step_days
            time = time.to(device)
            logits = init + rate * time[:, None, None]

        # Observe sequences via a multinomial likelihood.
        with time_plate, place_plate:
            pyro.sample(
                "obs",
                dist.Multinomial(logits=logits.unsqueeze(-2), validate_args=False),
                obs=counts.unsqueeze(-2),
            )

    def fit(self, lr=0.0001, num_steps=2000, log_every=250):
        pyro.clear_param_store()  # clear parameters from previous runs
        pyro.set_rng_seed(20211214)

        guide = AutoNormal(self.model, init_scale=0.01)
        optim = Adam({"lr": lr})
        svi = SVI(self.model, guide, optim, loss=Trace_ELBO(max_plate_nesting=3,ignore_jit_warnings=True))

        # Train (i.e. do ELBO optimization) for num_steps iterations
        losses = []
        for step in range(num_steps):
            loss = svi.step()
            losses.append(loss)

        def get_conditionals(data):
            trace = poutine.trace(poutine.condition(self.model, data)).get_trace()
            return {
                name: site["value"].detach()
                for name, site in trace.nodes.items()
                if site["type"] == "sample" and not site_is_subsample(site)
                if not name.startswith("obs")
            }

        result: dict = defaultdict(dict)
        rate = torch.zeros([len(locations), len(lineages)]).to(device)
        init = torch.zeros([len(locations), len(lineages)]).to(device)
        coef = torch.zeros([features.shape[1]]).to(device)
        coef_std = torch.zeros([features.shape[1]]).unsqueeze(0).to(device)
        rate_std = torch.zeros([len(locations), len(lineages)]).unsqueeze(0).to(device)
        init_std = torch.zeros([len(locations), len(lineages)]).unsqueeze(0).to(device)
        for i in range(1000):
            samples = get_conditionals(guide())
            rate = rate + samples['rate']
            init = init + samples['init'].to(device)
            coef = coef + samples['coef'].to(device)
            coef_std = torch.cat((coef_std, samples["coef"].unsqueeze(0).to(device)), 0)
            rate_std = torch.cat((rate_std, samples["rate"].unsqueeze(0).to(device)), 0)
            init_std = torch.cat((init_std, samples["init"].unsqueeze(0).to(device)), 0)
        rate, init, coef = rate.to(device), init.to(device), coef.to(device)
        result['mean']['rate'] = rate / 1000
        result['mean']['init'] = init / 1000
        result['mean']['coef'] = coef / 1000

        coef_std = coef_std[1:, :]
        result['std']['coef'] = ((coef_std - result['mean']['coef'].unsqueeze(0))**2).sum(0) / 1000
        result['std']['rate'] = ((rate_std - result['mean']['rate'].unsqueeze(0))**2).sum(0) / 1000
        result['std']['init'] = ((init_std - result['mean']['init'].unsqueeze(0))**2).sum(0) / 1000

        torch.save(result, './result.pkl')

