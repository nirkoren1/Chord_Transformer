import pickle

import pandas as pd
import os
import re
import pychord
from typing import List

file_path = r"C:\Users\Nirkoren\Desktop\chordData\chords_and_lyrics\chords_and_lyrics.csv"


def fix_chord(c: str):
    c = c.replace("dim", "")
    c = c.replace("sus", "")
    c = c.replace(")", "")
    c = c.replace("(", "")
    c = c.replace("+", "")
    c = c.replace("*", "")
    c = c.replace("-", "")
    if c.count("1") > 2:
        c.replace("11", "")
    if c.count("9") >= 2:
        c.replace("9", "")
    if c.count("4") >= 2:
        c.replace("4", "")
    if c.count("2") >= 2:
        c.replace("2", "")
    if c.count("7") >= 2:
        c.replace("7", "")
    return c


def get_chords(line):
    tmp = []
    out = []
    chord_regex = "\(?([ABCDEFG])([#b]?)(m?)-?(\(?[245679]?\)?)(\-?)(/?)((dim)|(sus)|(maj)|(aug)|)(\+?)(add)?(\(?([245679]|11|13)?\)?)M?(\*?)((/[ABCDEFG][#b]?)?)(\(hold\))?\)?"
    matches = re.findall(chord_regex, line)
    for match in matches:
        chord = ""
        for c in match:
            chord += c
        tmp.append(chord)
    i = 0
    while i < len(tmp):
        if (tmp[i][-1] == "/") and i != len(tmp) - 1:
            chord = fix_chord(tmp[i] + tmp[i+1])
            i += 1
            try:
                pychord.Chord(chord)
            except Exception as e:
                return None
            else:
                out.append(chord)
        else:
            chord = fix_chord(tmp[i])
            i += 1
            try:
                pychord.Chord(chord)
            except Exception as e:
                return None
            else:
                out.append(chord)
    return out


def get_all_chords(col):
    out = []
    for idx, chords in enumerate(col):
        print("\r", f"{idx}/{len(col)}", end="")
        out.append(get_chords(chords))
    return out


def parse_genres(genres):
    out = []
    for row in genres:
        genre_partition = []
        genre_regex = r"\w[\w \-&]+"
        matches = re.findall(genre_regex, row)
        for match in matches:
            genre = ""
            for c in match:
                genre += c
            genre_partition.append(genre)
        out.append(genre_partition)
    return out


def generate_training_data(df_: pd.DataFrame):
    out = []
    for i, row in df_.iterrows():
        out.append(row["genres"] + ["<start>"] + row["chords"] + ["<end>"])
    return out


if __name__ == '__main__':
    # prepare data
    df = pd.read_csv(file_path)
    out_df = pd.DataFrame()
    out_df["genres"] = parse_genres(df["genres"])
    out_df["popularity"] = df["popularity"]
    out_df["chords"] = get_all_chords(df["chords"])
    out_df = out_df.dropna(axis=0, how='any')
    out_df["training_data"] = generate_training_data(out_df)
    print(out_df.head())
    out_df.to_pickle("data.pickle")

    # prepare 1hot encoding
    df = pd.read_pickle("data.pickle")
    s = set()
    for index, row in df.iterrows():
        for chord in row["training_data"]:
            s.add(chord)
        print("\r", index)
    df_chords = pd.DataFrame()
    df_chords["chords"] = list(s)
    num_of_chords = len(df_chords["chords"])
    df_chords["one_hot"] = [[1 if j == i else 0 for j in range(num_of_chords)] for i in range(num_of_chords)]
    df_chords.to_pickle("chords1Hot.pickle")
    chords = pd.read_pickle("chords1Hot.pickle")
    print(chords)
