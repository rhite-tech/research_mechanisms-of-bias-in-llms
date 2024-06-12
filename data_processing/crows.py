import pandas as pd

# Read dataset
crows = pd.read_csv("crows-pairs/crows_pairs_anonymized.csv", index_col=0)

# Extract only gender bias (bias_type = 'gender')
gender_data = crows[crows['bias_type'] == 'gender']
print(gender_data)

# Save gender data as csv
gender_data.to_csv("crows-pairs/gender_crows_pairs.csv")