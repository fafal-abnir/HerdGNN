from typing import Literal

from datasets.DGraphFin import DGraphFin
from datasets.bitcoin_otc import BitcoinOTC
from datasets.mooc import MOOC
from datasets.reddit_body import RedditBody
from datasets.reddit_title import RedditTtile
from datasets.ellipticpp import EllipticPP


def get_dataset(root: str = "./data",
                name: Literal[
                    "EllipticPP", "DGraphFin", "BitcoinOTC", "MOOC", "RedditTitle", "RedditBody"] = "BitcoinOTC",
                edge_window_size: Literal["day", "week", "month"] = "week", num_windows: int = 30, force_reload=True):
    if name == "DGraphFin":
        return DGraphFin(root=f"{root}/DGraphFin", edge_window_size=edge_window_size, num_windows=num_windows,
                         force_reload=force_reload)
    if name == "EllipticPP":
        return EllipticPP(root=f"{root}/EllipticPP", num_windows=num_windows, force_reload=force_reload)
    elif name == "BitcoinOTC":
        return BitcoinOTC(root=f"{root}/BitcoinOTC", edge_window_size=edge_window_size, num_windows=num_windows,
                          force_reload=force_reload)
    elif name == "MOOC":
        return MOOC(root=f"{root}/MOOC")
    elif name == "RedditBody":
        return RedditBody(root=f"{root}/RedditBody", edge_window_size=edge_window_size, num_windows=num_windows,
                          force_reload=force_reload)
    elif name == "RedditTitle":
        return RedditTtile(root=f"{root}/RedditTitle", edge_window_size=edge_window_size, num_windows=num_windows,
                           force_reload=force_reload)
    else:
        raise RuntimeError(f"Wrong dataset name:{name}")
