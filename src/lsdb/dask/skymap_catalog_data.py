
from typing import Callable
import healpy as hp
import numpy as np
import pandas as pd
import dask.array as da


def skymap_agg(df, col, f=np.mean):
    assert all(df["order_pix"].values == df["order_pix"].iloc[0])
    ret_dict = {
        "pix" : [df["order_pix"].iloc[0]],
        "val" : [f(df[col].values)]
    }
    return pd.DataFrame(ret_dict, columns=ret_dict.keys())

def skymap_catalog_data(cat, col: str=None, order: int=6, func: Callable=np.mean):
    nside = hp.order2nside(order)
    ddf = cat._ddf.assign(order_pix =  lambda x: hp.ang2pix(nside, x["ra"], x["dec"], lonlat=True, nest=True))
    meta = {"pix": "i8", "val": "f8"}
    tf = ddf.groupby("order_pix").apply(skymap_agg, col, meta=meta).compute()

    npix = hp.order2npix(order)
    img = np.zeros(npix)
    img[tf["pix"].values] = tf["val"].values
    return img