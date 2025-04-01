from caveclient import CAVEclient
from standard_transform import minnie_transform_vx

import numpy as np
import pandas as pd


def download_nucleus_data(client, path2download, tables2download):

    for table in tables2download:
        print(table)
        auxtable = client.materialize.query_table(table)
        auxtable = pd.DataFrame(auxtable)
        auxtable.to_csv(f"{path2download}/{table}.csv")