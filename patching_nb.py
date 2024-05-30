import plotly.express as px
import json
from nnsight import LanguageModel
import json
import numpy as np
import sys

experiment_name = str(sys.argv[1])
# model = LanguageModel("/scratch-shared/tpungas/llama-13b")
model = LanguageModel("meta-llama/Meta-Llama-3-70B")

with open('experimental_outputs/patching/{}.json'.format(experiment_name), 'r') as f:
    out = json.load(f)[-1]
false_prompt = out['false_prompt']
logit_diffs = out['logit_diffs']
n_toks = len(logit_diffs)
model_name = out['model']

# Transpose logit_diffs
logit_diffs = [[logit_diffs[i][j] for i in range(0, len(logit_diffs))[::-1]] for j in range(len(logit_diffs[0]))]
probs = [[1 / (1 + np.exp(-logit)) for logit in layer] for layer in logit_diffs]

# Find smallest and largest diff for plot scale
all_diffs = [d for sublist in logit_diffs for d in sublist]
print("Min diff is {} and max diff is {}".format(min(all_diffs), max(all_diffs)))

token_ids = model.tokenizer(false_prompt)['input_ids']
tokens = [
    model.tokenizer.decode([token_id]) + f" ({idx})" for idx, token_id in enumerate(token_ids)
]
tokens = tokens[-n_toks:]

# Save the information to make tweaking the plot easier
# print("Tokens:", tokens)
# print("Logit diffs:", logit_diffs)

fig = px.imshow(
    logit_diffs,
    x=tokens,
    labels=dict(x="Token", y="Layer"),
    color_continuous_scale='blues',
    range_color=[-1.5, 3.5]  # Modify the range to be appropriate
)

# Change font size
fig.update_layout(
    font=dict(size=14)
)

fig.update_xaxes(tickangle=30)

fig.write_image("figures/{}.png".format(experiment_name), scale=3)