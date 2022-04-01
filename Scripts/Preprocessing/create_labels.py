"""
This module will create a CSV file will properly formatted multihot label rows suitable for ML, using the
given RSNA label file (which is badly formatted).
"""

import pandas as pd

in_file_path = '~/rsna-intracranial-hemorrhage-detection/stage_2_train.csv'
out_file_path = '~/rsna-intracranial-hemorrhage-detection/train_labels.csv'
df = pd.read_csv(in_file_path)

# Extract the type of hemorrhage from the ID column
df['type'] = df['ID'].str.split('_').str[2]
#also modify the 'ID' column to remove the type of hemorrhage from it
df['ID'] = df['ID'].str.split('_').str[:-1].str.join('_')
df.head(10)


# Add boolean columns for each type of hemorrhage (including 'any')
df['any'] = df[['Label','type']].apply(lambda x: x['Label'] == 1 and x['type'] == 'any', axis=1)
df['epidural'] = df[['Label','type']].apply(lambda x: x['Label'] == 1 and x['type'] == 'epidural', axis=1)
df['intraparenchymal'] = df[['Label','type']].apply(lambda x: x['Label'] == 1 and x['type']=='intraparenchymal', axis=1)
df['intraventricular'] = df[['Label','type']].apply(lambda x: x['Label'] == 1 and x['type']=='intraventricular', axis=1)
df['subarachnoid'] = df[['Label','type']].apply(lambda x: x['Label'] == 1 and x['type']=='subarachnoid', axis=1)
df['subdural'] = df[['Label','type']].apply(lambda x: x['Label'] == 1 and x['type']=='subdural', axis=1) 

# Group by each image ID to collapse each row representing an image into one
groups = df.groupby('ID')

labels = ['ID', 'Label', 'type', 'any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']


def collapse(group):
    data = sum(group.iloc[j].to_numpy()[3:].astype(bool) for j in range(len(group))).astype(bool).tolist()
    data = group.iloc[0].to_numpy()[:3].tolist() + data
    return pd.Series(data, index=labels)

df = groups.apply(collapse)

# Drop redundant columns
df = df.drop(['Label', 'type', 'ID'], axis='columns')
df.head(10)


# ## Save the formatted data for later use
df.to_csv(out_file_path)



# Some basic exploration

df['any'].value_counts()

# We see that there is a heavy imbalance in favor of 'no hemorrhage'. 
# Thus, we will may need to do some up/down sampling to rectify this during training.


types = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
for t in types:
    print(df[t].value_counts())


# We can also see an imbalance among the 5 positive classes. Most evident is 'epidural', which only has 3145 images containing it.
