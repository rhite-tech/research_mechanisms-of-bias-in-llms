import pandas as pd

def names_to_pronouns(sent):
    # Ignore these specific examples
    if sent.startswith("Her brother") or sent.startswith("Her sister") or sent.startswith('"Yes') or sent.startswith("His mind wondered") or sent.startswith("My cousin") or sent.startswith("James and his roommate") or sent.startswith("The detective told") or sent.startswith("My friend") or sent.startswith("Emma made") or "his body guard" in sent or "harsh and ruthless" in sent:
        return sent

    if sent.startswith("The task was trivial") or sent.startswith("I'm worried about"):
        sent.replace("James", "him")
        sent.replace("Olivia", "her")
        return sent

    # Beginning of the sentence
    if sent.startswith("Olivia's"):
        sent = sent.replace("Olivia's", "Her")
    elif sent.startswith("James'"):
        sent = sent.replace("James'", "His")
    elif sent.startswith("Olivia"):
        sent = sent.replace("Olivia", "She")
    elif sent.startswith("James"):
        sent = sent.replace("James", "He")

    # James' and Olivia's inside sentence
    if "James'" in sent:
        sent = sent.replace("James'", "his")
    if "Olivia's" in sent:
        sent = sent.replace("Olivia's", "her")

    # Other mentions
    if "James" in sent:
        sent = sent.replace("James", "he")
    if "Olivia" in sent:
        sent = sent.replace("Olivia", "she")

    return sent


# Read dataset
crows = pd.read_csv("crows-pairs/cleaned_cps.csv", index_col=0)
crows = crows.drop(["A_x", "B_x", "stereo_antistereo"], axis=1)

# Replace names with pronouns (Olivia => she/her, James => he/his)
crows['A_en'] = crows['A_en'].map(lambda sent: names_to_pronouns(sent))
crows['B_en'] = crows['B_en'].map(lambda sent: names_to_pronouns(sent))

# print(crows.to_string())

# Creating separate dataframes for stereotypes and anti-stereotypes
stereotype_df = crows[['A_en']].rename(columns={'A_en': 'statement'})
stereotype_df['label'] = 1

less_stereotype_df = crows[['B_en']].rename(columns={'B_en': 'statement'})
less_stereotype_df['label'] = 0

# Concatenating the two dataframes
result_df = pd.concat([stereotype_df, less_stereotype_df])

# Resetting the index of the concatenated dataframe
result_df = result_df.reset_index(drop=True)

# Shuffle the rows
shuffled = result_df.sample(frac=1)
print(shuffled.to_string())

# Save data as csv
# shuffled.to_csv("crows-pairs/experiment_cps.csv", index=False)
