import torch

def train_dataset(t):
    dataset = torch.load('./data/Omicron_experimental_data_time.pkl')

    counts = dataset['counts'][:t,:,:]
    time_step_days = dataset["time_step_days"]
    features = dataset['mutation_features'][:t,:,:].sum(0)/counts[:t,:,:].sum(0).sum(0).unsqueeze(-1)
    affinity = dataset['affinity_features'][:t,:,:,:].sum(0)/counts[:t,:,:].sum(0).sum(0).unsqueeze(-1).unsqueeze(-1)
    escape = dataset['escape_features'][:t,:,:,:].sum(0)/counts[:t,:,:].sum(0).sum(0).unsqueeze(-1).unsqueeze(-1)
    features = torch.where(torch.isnan(features), torch.full_like(features, 0), features)
    affinity = torch.where(torch.isnan(affinity), torch.full_like(affinity, 0), affinity)
    escape = torch.where(torch.isnan(escape), torch.full_like(escape, 0), escape)
    place_lineage_index = (counts.ne(0).any(0).reshape(-1).nonzero(as_tuple=True)[0])

    locations = dataset['locations']
    lineages = dataset['lineages']
    mutations = dataset['mutations']

    train_dataset = {
        'counts': counts,
        'time_step_days': time_step_days,
        'features': features,
        'affinity': affinity,
        'escape': escape,
        'plcae_lineage_index': place_lineage_index,
        'locations': locations,
        'lineages': lineages,
        'mutations': mutations
    }

    return train_dataset


def total_dataset():
    dataset = torch.load('./data/Omicron_experimental_data_new.pkl')

    counts = dataset['counts']
    features = dataset['mutation_features']
    affinity = dataset['affinity_features']
    escape = dataset['escape_features']
    time_step_days = dataset["time_step_days"]

    locations = dataset['locations']
    lineages = dataset['lineages']
    mutations = dataset['mutations']
    place_lineage_index = (counts.ne(0).any(0).reshape(-1).nonzero(as_tuple=True)[0])

    train_dataset = {
        'counts': counts,
        'time_step_days': time_step_days,
        'features': features,
        'affinity': affinity,
        'escape': escape,
        'plcae_lineage_index': place_lineage_index,
        'locations': locations,
        'lineages': lineages,
        'mutations': mutations
    }

    return train_dataset