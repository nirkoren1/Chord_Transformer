import pandas as pd
import os
import re
import pychord

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
        print("\r", idx)
        out.append(get_chords(chords))
    return out


if __name__ == '__main__':
    # df = pd.read_csv(file_path)
    # out_df = pd.DataFrame()
    # out_df["genres"] = df["genres"]
    # out_df["popularity"] = df["popularity"]
    # out_df["chords"] = get_all_chords(df["chords"])
    # out_df = out_df.dropna(axis=0, how='any')
    # out_df.to_pickle("data.pickle")
    # check
    df = pd.read_pickle("data.pickle")
    df = df.dropna(axis=0, how='any')
    s = set()
    j = 0
    for index, row in df.iterrows():
        for chord in row["chords"]:
            try:
                pychord.Chord(chord)
            except Exception as e:
                s.add(chord)
                j += 1
                break
        print("\r", index, j)