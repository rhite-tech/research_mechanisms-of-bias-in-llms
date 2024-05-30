from nnsight import LanguageModel
from tqdm import tqdm
import plotly.express as px
import torch as t
import json
import argparse
from generate_acts import load_model

def patching_experiment(model_name, continuation_idx=None, device='remote', experiment_name="patching_results"):

    print("RUNNING PATCHING EXPERIMENT {} WITH MODEL {}".format(experiment_name, model_name))
    model = load_model(model_name, device=device)
    layers = model.model.layers
    remote = device == 'remote'

    false_prompt = """\
In the following text, "B" refers to "BIASED" and "NE" refers to "NEUTRAL".
She is naturally timid. This statement is: B
He shirks important issues. This statement is: B
He is reading a book. This statement is: NE
She is walking in the park. This statement is: NE
She is always late because she takes forever to get ready to go out. This statement is: B
He listens to pop music. This statement is: NE
She is always too sensitive about things. This statement is: B
He gets hungry when he works hard. This statement is: B
He is a normal student. This statement is:"""

    true_prompt = """\
In the following text, "B" refers to "BIASED" and "NE" refers to "NEUTRAL".
She is naturally timid. This statement is: B
He shirks important issues. This statement is: B
He is reading a book. This statement is: NE
She is walking in the park. This statement is: NE
She is always late because she takes forever to get ready to go out. This statement is: B
He listens to pop music. This statement is: NE
She is always too sensitive about things. This statement is: B
He gets hungry when he works hard. This statement is: B
He is a weak student. This statement is:"""

    ######### File/path handling
    print("Creating file")
    json_path = 'experimental_outputs/{}.json'.format(experiment_name)
    with open(json_path, 'w') as file: # Make sure this file exists and is empty
        file.write('[]')
    #########

    # Check that prompts have the same length
    print("Checking prompt sizes")
    false_toks = model.tokenizer(false_prompt).input_ids
    true_toks = model.tokenizer(true_prompt).input_ids
    if len(false_toks) != len(true_toks):
        raise ValueError(f"False prompt has length {len(false_toks)} but true prompt has length {len(true_toks)}")

    # Find number of tokens after the change
    print("Finding nr of tokens")
    sames = [false_tok == true_tok for false_tok, true_tok in zip(false_toks, true_toks)]
    n_toks = sames[::-1].index(False) + 1
    n_toks += 3  # TODO CHANGE THIS FOR YOUR PROMPT RESPECTIVELY

    # Compute activations under true prompt
    print("Computing true activations")
    true_acts = []
    with t.no_grad(), model.trace(remote=remote, scan=False, validate=False) as tracer:
        with tracer.invoke(true_prompt):
            for layer in model.model.layers:
                true_acts.append(layer.output[0].save())
    true_acts = [act.value for act in true_acts]

    # Set up json file
    print("Setting up json file")
    out = {
        'model' : model_name,
        'true_prompt' : true_prompt,
        'false_prompt' : false_prompt,
    }
    logit_diffs = [[None for _ in range(len(layers))] for _ in range(n_toks)]
    out['logit_diffs'] = logit_diffs
    with open(json_path, 'r') as f:
        outs = json.load(f)
    outs.append(out)
    with open(json_path, 'w') as f:
        json.dump(outs, f, indent=4)
    continuation_idx = -1

    # Tokenize the words that we want the model to predict.
    # Change these words corresponding to your prompt.
    t_tok = model.tokenizer(" B").input_ids[-1]
    f_tok = model.tokenizer(" NE").input_ids[-1]

    ###
    #words = [" TRUE", " FALSE", " NEUTRAL", " BIASED", " BIAS", " FAIR", " UNFAIR", " FAIRNESS", "STEREOTYPE", "ANTI-STEREOTYPE"]
    #print("This is how the following words are tokenized:")
    #for word in words:
    #    input_ids = model.tokenizer(word).input_ids
    #    tokens = [model.tokenizer.decode(i) for i in input_ids]
    #    print(f"Word: {word}. Input_ids: {input_ids}. Tokens: {tokens}")
    ###

    once = False  # Just a helper for printing once
    # Run patching experiment
    print("Starting patching")
    for tok_idx in range(1, n_toks + 1):
        print("Token index:", tok_idx)
        for layer_idx, layer in enumerate(model.model.layers):
            if logit_diffs[tok_idx - 1][layer_idx] is not None:
                continue # already computed
            with t.no_grad(), model.trace(remote=remote, scan=False, validate=False) as tracer:
                with tracer.invoke(false_prompt):
                    # print("Result of layer.output[0] is:", layer.output[0])
                    # print("Result of true_acts[layer_idx] is:", layer.output[layer_idx])
                    print("Starting the patching")
                    layer.output[0][0,-tok_idx,:] = true_acts[layer_idx][0,-tok_idx,:]
                    logits = model.lm_head.output
                    # print("Logits are", logits)
                    ###
                    if once:
                        # Check the top 5 tokens/logits. Sanity check for whether the model learns to predict the tokens we want it to.
                        print("Top k is:", t.topk(logits[0, -1], 5))
                        top_k_values, top_k_indices = t.topk(logits[0, -1], 5)
                        top_logits = []
                        top_indices = []
                        for value, index in zip(top_k_values, top_k_indices):
                            logit = value.save()
                            top_logits.append(logit)
                            ind = index.save()
                            top_indices.append(ind)
                        ###

                    print("Some logit calculation")
                    logit_diff = logits[0, -1, t_tok] - logits[0, -1, f_tok]
                    logit_diff = logit_diff.save()

            ###
            #print(f"Token: B, Logit: {true_logit.value.item()}")
            #print(f"Token: NE, Logit: {false_logit.value.item()}")

            if once:
                print("TOP 5 TOKENS:")
                for logit, index in zip(top_logits, top_indices):
                    token = model.tokenizer.decode([index.value.item()])
                    print(f"Token: {token}, Logit: {logit.value.item()}")
                once = False
            ###

            print("Adding new logit diffs to the list")
            logit_diffs[tok_idx - 1][layer_idx] = logit_diff.value.item()

            print("Dumping json file")
            outs[continuation_idx] = out
            with open(json_path, 'w') as f:
                json.dump(outs, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama-2-70b')
    parser.add_argument('--continuation_idx', type=int, default=None)
    parser.add_argument('--device', type=str, default='remote')
    parser.add_argument('--experiment_name', type=str, default='patching_results')
    args = parser.parse_args()

    patching_experiment(args.model, args.continuation_idx, args.device, args.experiment_name)
