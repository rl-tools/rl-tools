import itertools
import json

sweep = {
    "dmodel": [16],
    "seed": list(range(5)),
    "num_teachers:num_episodes": [(1000, 10), (500, 20), (250, 40), (125, 80), (64, 156), (32, 312), (16, 624)],
}

def cartesian_product(sweep):
    import itertools
    keys = sweep.keys()
    values = (sweep[key] for key in keys)
    for instance in itertools.product(*values):
        yield dict(zip(keys, instance))

for job in cartesian_product(sweep):
    output = {}
    for k, v in job.items():
        if isinstance(v, tuple):
            output[k.split(":")[0]] = v[0]
            output[k.split(":")[1]] = v[1]
        else:
            output[k] = v
    print(f"{json.dumps(output)}")
