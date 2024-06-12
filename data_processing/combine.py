import pandas as pd
import re

def consistent_format(sent):
    new_sentence = sent.rstrip()
    new_sentence = new_sentence.lstrip()
    new_sentence = re.sub(r'\s+', ' ', new_sentence)
    new_sentence = new_sentence.capitalize()
    if new_sentence[-1] not in ['.', '!', '?']:
        new_sentence += '.'
    ' '.join(new_sentence.split())
    return new_sentence


# Choose either inter- or intrasentence
SENTENCE_TYPE = "intersentence"
bias_types = ['race', 'gender', 'profession', 'religion']
bias_type = 'race'

# Read datasets
dev = pd.read_csv("stereoset/{}/{}_dev_{}_stereoset.csv".format(SENTENCE_TYPE, SENTENCE_TYPE[:5], bias_type))
test = pd.read_csv("stereoset/{}/{}_test_{}_stereoset.csv".format(SENTENCE_TYPE, SENTENCE_TYPE[:5], bias_type))

# Combine dataframes
combined = pd.concat([dev, test], ignore_index=True)

# Remove unnamed column
combined = combined.loc[:, ~combined.columns.str.contains('^Unnamed')]

# print(combined.to_string())

# Save combined dataframe
combined.to_csv("stereoset/{}/{}_combined_{}_stereoset.csv".format(SENTENCE_TYPE, SENTENCE_TYPE[:5], bias_type), index=False)

# Format data for geometry of truth experiments
# Label 1 means a stereotype, label 0 means an anti-stereotype
geo_df = combined.drop(["target", "unrelated"], axis=1)
geo_df['context'] = geo_df['context'].map(lambda sent: consistent_format(sent))
# print(geo_df.to_string())

# Prepend context to stereotype and anti-stereotype and drop context column
if SENTENCE_TYPE == "intersentence":
    geo_df['stereotype'] = "Context: " + geo_df['context'] + '\nStatement: ' + geo_df['stereotype']
    geo_df['anti_stereotype'] = "Context: " + geo_df['context'] + '\nStatement: ' + geo_df['anti_stereotype']
geo_df = geo_df.drop("context", axis=1)
# print(geo_df.to_string())

# Creating separate dataframes for stereotypes and anti-stereotypes
stereotype_df = geo_df[['stereotype']].rename(columns={'stereotype': 'statement'})
stereotype_df['label'] = 1

anti_stereotype_df = geo_df[['anti_stereotype']].rename(columns={'anti_stereotype': 'statement'})
anti_stereotype_df['label'] = 0

# Concatenating the two dataframes
result_df = pd.concat([stereotype_df, anti_stereotype_df])

# Resetting the index of the concatenated dataframe
result_df = result_df.reset_index(drop=True)

# Shuffle the rows
shuffled = result_df.sample(frac=1)
# print(shuffled.to_string())

# Save data as csv
shuffled.to_csv("stereoset/{}/experiment_{}_{}_stereoset.csv".format(SENTENCE_TYPE, SENTENCE_TYPE[:5], bias_type), index=False)
