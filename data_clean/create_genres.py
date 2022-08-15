import csv
import os
import pandas as pd
import re


genres = []
df_one_hot = pd.read_pickle(r"C:\Users\Nirkoren\PycharmProjects\Chord_Transformer\data_clean\chords1Hot.pickle")
chord_regex = "\(?([ABCDEFG])([#b]?)(m?)-?(\(?[245679]?\)?)(\-?)(/?)((dim)|(sus)|(maj)|(aug)|)(\+?)(add)?(\(?([245679]|11|13)?\)?)M?(\*?)((/[ABCDEFG][#b]?)?)(\(hold\))?\)?"
for idx, row in df_one_hot.iterrows():
    if not re.match(chord_regex, row["chords"]):
        genres.append(row["chords"])

genres.sort()

with open("genres.txt", 'a') as f:
    for genre in genres:
        print(genre, file=f)
