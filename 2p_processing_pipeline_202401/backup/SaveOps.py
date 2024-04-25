#!/usr/bin/env python3

import json
from suite2p import default_ops

ops = default_ops()

with open("config.json", "w") as f: 
    json.dump(ops, f)