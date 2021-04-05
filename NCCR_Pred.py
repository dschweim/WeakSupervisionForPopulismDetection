import os
import glob
import pandas as pd

# Set working directory
os.chdir(r"C:\Users\dschw\Documents\GitHub\Thesis\Data\NCCR_Content\NCCR_Content\Texts\Texts")

# Get filenames of texts
files = [i for i in glob.glob("*.txt")]

# Remove English files
files = [x for x in files if not x.startswith('uk')]

# Create empty dataframe for texts
df = pd.DataFrame()

# Read every txt file in dir and add to df
for file in files:
    tmp = open(file,  encoding="ISO-8859-1").read()
    txt = pd.DataFrame({'id': [file],
                        'text': [tmp]})
    df = df.append(txt)

# Cut off unnecessary info from text strings
#tbd regex

## Add Populism labels as labelled in Text_Table.txt
#table_text = pd.read_csv(f'C:/Users/dschw/Documents/GitHub/Thesis/Data/NCCR_Content/NCCR_Content', delimiter = "\t", encoding = "ISO-8859-1")
#print(list(table_text))


# Save created corpus
df.to_csv(f'C:/Users/dschw/Documents/GitHub/Thesis/nccr_corpus.csv', index=False)

print(df)

