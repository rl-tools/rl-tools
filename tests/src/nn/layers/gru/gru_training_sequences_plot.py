import json
import matplotlib.pyplot as plt
import code

with open('gru_training_sequences.json') as f:
    data = json.load(f)
    # code.interact(local=dict(globals(), **locals()))
    n_plots = 5
    fig, axs = plt.subplots(n_plots, 1, figsize=(10, 10))
    fig.suptitle(f"{json.dumps(data['meta'])}")
    for i in range(n_plots):
        batch = data["batch"]
        ax = axs[i]
        print(f"data: {[s['target'] for s in batch[i]]}")
        label_set_input = False
        label_set_reset = False
        for j, s in enumerate(batch[i]):
            if s["input"] == 1:
                ax.axvline(x=j, color='black', alpha=0.2, linestyle='--', label="input" if label_set_input == False else None)
                label_set_input = True
            if s["reset"] == 1:
                ax.axvline(x=j, color='r', alpha=0.5, linestyle='-', label="reset" if label_set_reset == False else None)
                label_set_reset = True
        ax.plot(range(len(batch[i])), [s["target"] for s in batch[i]], label="target")
        ax.plot(range(len(batch[i])), [s["output"] for s in batch[i]], label="output")
        # vertical line if s["input"] is 1
        ax.legend()
    plt.show()
