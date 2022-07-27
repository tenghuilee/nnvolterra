from io import TextIOWrapper
import json
import numpy as np
import time
import torch

JLogDict = dict()
JLogDict["date"] = time.asctime(time.localtime())


def record_append(tag: str, info: str, **kwargs):
    print("\033[31m record appended\033[34m", tag, "\033[0m")
    # print("info", info, "\033[0m")

    JLogDict[tag] = {
        "info": info,
        **kwargs,
    }


class _jsdec(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, torch.Tensor):
            return obj.data.cpu().numpy().tolist()
        return super().default(obj)


def record_dump_to_file(file):
    if isinstance(file, TextIOWrapper):
        dest = file
        json.dump(JLogDict, file, cls=_jsdec, indent=2)
    elif isinstance(file, str):
        if file.endswith(("JSON", "json")):
            dest = open(f"result/{file}", "w")
        else:
            dest = open(f"result/{file}.json", "w")
        json.dump(JLogDict, dest, cls=_jsdec, indent=2)
        dest.close()

    print("\033[31m record dumped to\033[34m", dest.name, "\033[0m")

def record_load_file(file):
    with open(f"result/{file}.json", "r") as rec:
        ans = json.load(rec)
    return ans
