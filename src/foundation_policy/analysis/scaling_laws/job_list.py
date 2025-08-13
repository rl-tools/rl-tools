import itertools
import json

sweep = {
    "dmodel": [8, 16, 32, 64, 128],
    "seed": list(range(10))
}

def cartesian_product(sweep):
    import itertools
    keys = sweep.keys()
    values = (sweep[key] for key in keys)
    for instance in itertools.product(*values):
        yield dict(zip(keys, instance))

for job in cartesian_product(sweep):
    print(f"{json.dumps(job)}")
