import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import torch
import datetime

df_affinity = pd.read_csv('./data/affinity_Omicron.csv', header = 0)

dic_affinity = {
    "Mutation" : df_affinity['mutation'],
    "Affinity" : df_affinity['bind_avg']
}
experimental_df_affinity = pd.DataFrame(dic_affinity)


df_escape = pd.read_csv('./data/escape_preOmicron.csv', header = 0)

df_escape['site'] = ''
for i in range(len(df_escape)):
    df_escape['site'][i] = df_escape['mutation'][i][1:-1]
df_escape['site'] = df_escape['site'].apply(pd.to_numeric)

experimental_df_escape = df_escape.drop(df_escape[(df_escape['site']>531) | (df_escape['site']<331)].index) 

# process affinity & escape data
# get affinity/escape index
def get_loc(i, list):
    for index, value in enumerate(list):
        if i < value:
            return index
    else:
        return index + 1

bin = 400

affinity_range = []
escape_range = []

for i in range(bin + 1):
    affinity_range.append(experimental_df_affinity['Affinity'].min() + i * (experimental_df_affinity['Affinity'].max() - experimental_df_affinity['Affinity'].min()) / bin)
    escape_range.append(experimental_df_escape['escape'].min() + i * (experimental_df_escape['escape'].max() - experimental_df_escape['escape'].min()) / bin)



experimental_df = pd.read_csv('./Omicron_experimental_data.csv', header = 0, encoding='ISO-8859-1')
# experimental_df = pd.read_csv('/home/lzy/pyro_program/data/test2.csv', header = 0, encoding='ISO-8859-1')

print("\nmutation string to list process starts.")
experimental_df['AA Mutation'] = ''
for i in range(len(experimental_df['Mutation'])):
    experimental_df['AA Mutation'][i] = experimental_df['Mutation'][i].split(',')
    if i % 50000 == 0:
            print(f" step {i} completed.")
print("mutation string to list process completed.")


def parse_date(string):
        return datetime.datetime.strptime(string, "%Y-%m-%d")

# for i in range(len(experimental_df)):
#         try:
#             experimental_df["Time"][i] = parse_date(experimental_df["Time"][i])
#         except ValueError:
#             # skipped["date"] += 1
#             continue
# print("\n time(string to datetime) process completed.")

start_date = parse_date('2021-11-01')
print("\ncalculation of start_time completed.", start_date)

def main(experimental_df, experimental_df_affinity, experimental_df_escape, affinity_range, escape_range, start_date):

    
    print("\nCalculate the statistics of tensor process starts:")

    columns = defaultdict(list)
    stats = defaultdict(Counter)
    skipped = Counter()

    # Process rows one at a time.
    for i in range(len(experimental_df)):

        # Parse date.
        try:
            date = parse_date(experimental_df["Time"][i])
        except ValueError:
            skipped["date"] += 1
            continue
        day = (date - start_date).days

        # Parse location.
        location = experimental_df["Place"][i]
        if location in ("", "?"):
            skipped["Place"] += 1
            continue

        # Parse lineage.
        if type(experimental_df["Lineage"][i]) == str:
            lineage = experimental_df["Lineage"][i]
        else:
            continue
        # if lineage in ("Unassigned"):
        #     skipped["Lineage"] += 1
        #     continue

        # affinity & escape.
        for j in experimental_df['AA Mutation'][i]:
            site = j[1: 4]
            columns["site"].append(site)
            stats["site"][site] += 1

            index_1 = experimental_df_affinity[experimental_df_affinity.Mutation == j].index.tolist()
            if len(index_1) != 0:
                affinity_index = get_loc(experimental_df_affinity['Affinity'][index_1[0]], affinity_range)
                stats["affinity_index"][affinity_index] += 1
                stats['lineage_affinity_site'][lineage, affinity_index, site] += 1

            index_2 = experimental_df_escape[experimental_df_escape.mutation == j].index.tolist()
            if len(index_2) != 0:
                escape_index = get_loc(experimental_df_escape['escape'][index_2[0]], escape_range)
                stats["escape_index"][escape_index] += 1
                stats['lineage_escape_site'][lineage, escape_index, site] += 1

        # Append row.
        columns["day"].append(day)
        columns["location"].append(location)
        columns["lineage"].append(lineage)

        # Record stats.
        stats["day"][day] += 1
        stats["location"][location] += 1
        stats["lineage"][lineage] += 1
        for aa in experimental_df["AA Mutation"][i]:
            stats["aa"][aa] += 1
            stats["lineage_aa"][lineage, aa] += 1
        
        if i % 10000 == 0:
            print(f" step {i} completed.")
    print("Calculation of columns and stats completed.")

    columns = dict(columns)
    stats = dict(stats)
    print("\ndate min =", min(stats['day']))

    # Create contiguous coordinates.
    days = sorted(stats["day"])
    locations = sorted(stats["location"])
    lineages = sorted(stats["lineage"])
    affinity_indices = sorted(stats["affinity_index"])
    escape_indices = sorted(stats["escape_index"])
    sites = sorted(stats["site"])

    aa_counts = Counter()
    for (lineage, aa), count in stats["lineage_aa"].most_common():
        if count * 3 >= stats["lineage"][lineage]:
            aa_counts[aa] += count
    aa_mutations = [aa for aa, _ in aa_counts.most_common()]

    # aa_mutations = sorted(stats['aa'])

    # Create a dense feature matrix. X_{ML}
    aa_features = torch.zeros(len(lineages), len(aa_mutations), dtype=torch.float)
    for s, lineage in enumerate(lineages):
        for f, aa in enumerate(aa_mutations):
            count = stats["lineage_aa"].get((lineage, aa))
            if count is None:
                continue
            aa_features[s, f] = count / stats["lineage"][lineage]
    features = {
        "lineages": lineages,
        "aa_mutations": aa_mutations,
        "aa_features": aa_features,
    }
    print("X_ML completed.")

    # Create a dense feature matrix. X_{AL}
    affinity_features = torch.zeros(len(lineages), len(affinity_indices), len(sites), dtype=torch.float)
    for s, lineage in enumerate(lineages):
        for f, affinity_index in enumerate(affinity_indices):
            for l, site in enumerate(sites):
                count = stats["lineage_affinity_site"].get((lineage, affinity_index, site))
                if count is None:
                    continue
                affinity_features[s, f, l] = count / stats["lineage"][lineage]
    print("X_AL completed.")

    # Create a dense feature matrix. X_{EL}
    escape_features = torch.zeros(len(lineages), len(escape_indices), len(sites), dtype=torch.float)
    for s, lineage in enumerate(lineages):
        for f, escape_index in enumerate(escape_indices):
            for l, site in enumerate(sites):
                count = stats["lineage_escape_site"].get((lineage, escape_index, site))
                if count is None:
                    continue
                escape_features[s, f, l] = count / stats["lineage"][lineage]
    print("X_EL completed.")


    # Set time step
    time_step_days = 4

    # Create a dense dataset.
    location_id = {location: i for i, location in enumerate(locations)}
    lineage_id = {lineage: i for i, lineage in enumerate(lineages)}
    T = max(stats["day"]) // time_step_days + 1
    P = len(locations)
    S = len(lineages)
    counts = torch.zeros(T, P, S)
    for day, location, lineage in zip(
        columns["day"], columns["location"], columns["lineage"]
    ):
        t = day // time_step_days
        p = location_id[location]
        s = lineage_id[lineage]
        counts[t, p, s] += 1
    dataset = {
        "start_date": start_date,
        "time_step_days": time_step_days,
        "locations": locations,
        "lineages": lineages,
        "mutations": aa_mutations,
        "mutation_features": aa_features,
        "counts": counts,
        "affinity_features": affinity_features,
        "escape_features": escape_features
    }
    torch.save(dataset, './data/Omicron_experimental_data.pkl')

    ### Distribution of mutation for each time interval (truncated dataset for forecast)

    aa_features_time = torch.zeros(counts.shape[0], len(lineages), len(aa_mutations), dtype=torch.float)
    for s, lineage in enumerate(lineages):
        for f, aa in enumerate(aa_mutations):
            for d, day in enumerate(days):
                t = day // time_step_days
                count = stats["date_lineage_aa"].get((day, lineage, aa))
                if count is None:
                    continue
                aa_features_time[t, s, f] = aa_features_time[t, s, f] + count
    print("X_TLM completed.")
    
    affinity_features_time = torch.zeros(counts.shape[0], len(lineages), len(affinity_indices), len(sites), dtype=torch.float)
    for s, lineage in enumerate(lineages):
        for f, affinity_index in enumerate(affinity_indices):
            for l, site in enumerate(sites):
                for d, day in enumerate(days):
                    t = day // time_step_days
                    count = stats["date_lineage_affinity_site"].get((day, lineage, affinity_index, site))
                    if count is None:
                        continue
                    affinity_features_time[t, s, f, l] = affinity_features_time[t, s, f, l] + count
    print("X_TLAS completed.")

    escape_features_time = torch.zeros(counts.shape[0], len(lineages), len(escape_indices), len(sites), dtype=torch.float)
    for s, lineage in enumerate(lineages):
        for f, escape_index in enumerate(escape_indices):
            for l, site in enumerate(sites):
                for d, day in enumerate(days):
                    t = day // time_step_days
                    count = stats["date_lineage_escape_site"].get((day, lineage, escape_index, site))
                    if count is None:
                        continue
                    escape_features_time[t, s, f, l] = escape_features_time[t, s, f, l] + count
    print("X_TLES completed.")

    dataset1 = {
        "start_date": start_date,
        "time_step_days": time_step_days,
        "locations": locations,
        "lineages": lineages,
        "mutations": aa_mutations,
        "mutation_features": aa_features_time,
        "counts": counts,
        "affinity_features": affinity_features_time,
        "escape_features": escape_features_time
    }
    torch.save(dataset1, './data/Omicron_experimental_data_time.pkl')


if __name__ == "__main__":
    main(experimental_df, experimental_df_affinity, experimental_df_escape, affinity_range, escape_range, start_date)