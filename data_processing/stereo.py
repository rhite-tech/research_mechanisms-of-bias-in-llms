import pandas as pd
import json
import re

def consistent_format(sent):
    new_sentence = sent.rstrip()
    new_sentence = new_sentence.lstrip()
    new_sentence = re.sub(r'\s+', ' ', new_sentence)
    new_sentence = new_sentence.capitalize()
    if new_sentence[-1] not in ['.', '!', '?']:
        new_sentence += '.'
    return new_sentence


bias_types = ['race', 'gender', 'profession', 'religion']
bias_type = 'religion'
intra = True
split = 'test'  # dev or test

# Read dataset
with open("stereoset/raw/{}.json".format(split)) as f:
    stereo = json.load(f)

inter_gender_data = []
for tup in stereo['data']['intersentence']:
    if tup['bias_type'] == bias_type:
        new_tup = {key: tup[key] for key in ["target", "context", "sentences"]}
        for sentence in new_tup['sentences']:
            sentence.pop('id')
            sentence.pop('labels')
        inter_gender_data.append(new_tup)

intra_gender_data = []
for tup in stereo['data']['intrasentence']:
    if tup['bias_type'] == bias_type:
        new_tup = {key: tup[key] for key in ["target", "context", "sentences"]}
        for sentence in new_tup['sentences']:
            sentence.pop('id')
            sentence.pop('labels')
        intra_gender_data.append(new_tup)

# Convert json to dataframe
if intra:
    df = pd.json_normalize(intra_gender_data)  # , meta=['target', 'context', ['sentences', 'sentence']])
else:
    df = pd.json_normalize(inter_gender_data)  # , meta=['target', 'context', ['sentences', 'sentence']])

# Separate "sentences" into three columns: unrelated, stereotype and anti-stereotype
stereotypes = []
anti_stereotypes = []
unrelated = []
for row in df['sentences']:
    for sentence in row:
        if sentence['gold_label'] == "stereotype":
            stereotypes.append(consistent_format(sentence['sentence']))
        elif sentence['gold_label'] == "anti-stereotype":
            anti_stereotypes.append(consistent_format(sentence['sentence']))
        elif sentence['gold_label'] == "unrelated":
            unrelated.append(consistent_format(sentence['sentence']))
        else:
            print("WEIRD GROUND TRUTH LABEL:", sentence['gold_label'])

df['stereotype'] = stereotypes
df['anti_stereotype'] = anti_stereotypes
df['unrelated'] = unrelated

df = df.drop("sentences", axis=1)

# Show dataframe
print(df.to_string())

# Save dataframe
if intra:
    df.to_csv("stereoset/intrasentence/intra_{}_{}_stereoset.csv".format(split, bias_type))
else:
    df.to_csv("stereoset/intersentence/inter_{}_{}_stereoset.csv".format(split, bias_type))


