import itertools
import json

def cartesian_product(sweep):
    import itertools
    keys = sweep.keys()
    values = (sweep[key] for key in keys)
    for instance in itertools.product(*values):
        yield dict(zip(keys, instance))

def output(sweep):
    for job in cartesian_product(sweep):
        output = {}
        for k, v in job.items():
            if isinstance(v, tuple):
                output[k.split(":")[0]] = v[0]
                output[k.split(":")[1]] = v[1]
            else:
                output[k] = v
        print(f"{json.dumps(output)}")


sweep_dmodel = {
    "dmodel": [4, 8, 16, 32, 48, 64],
    "seed": list(range(10)),
    "num_teachers:num_episodes": [(1000, 10)],
    "teacher_selection": ["all"],
}

sweep_num_teachers = {
    "dmodel": [16],
    "seed": list(range(10)),
    "teacher_selection": ["random"],
    "num_teachers:num_episodes": [(1000, 10), (500, 20), (250, 40), (125, 80), (64, 156), (32, 312), (16, 624)],
}

output(sweep_dmodel)
output(sweep_num_teachers)