# RLtools Zoo

What is a zoo? A zoo contains species in the form of [Algorithm x Environment x Seed]. [Alogorithm x Environment] is given by a target in form of a `.cpp` file in `./{Algorithm}/{Environment}.cpp`. The seed is given by the `--seed` argument and the output path is given by `--output`. The RLtools Experiments signature is hence `{DATE}/{COMMIT_HASH}_zoo_algorithm_environment/{Algorithm}_{Environment}/{Seed}`. The `return.json` contains deterministic evaluation returns: 
```
{
    "steps": [0, 1000, ...],
    "returns": [
        [10.0, 09.0, ...],
        [20.0, 25.0, ...],
        ...
    ]
}
```
which allows for easy creation of learning curves.