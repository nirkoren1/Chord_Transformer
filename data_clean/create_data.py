import csv
import os
import pandas as pd


def valid_chord(chord):
    good_letters = ["A", "B", "C", "D", "E", "F", "G"]
    bad_letters = ["(", ")", "-", "'", '"']
    i = 0
    for letter in good_letters:
        if letter in chord:
            i += 1
    if i == 0:
        return False
    for letter in bad_letters:
        if letter in chord:
            return False
    return True


file_path = r"C:\Users\Nirkoren\Desktop\chordData"
chords_set = {"A"}
files = os.listdir(file_path)
for file in files:
    data = pd.read_csv(file_path + "\\" + file)
    for chords in data["Chords"]:
        try:
            for chord in chords.split():
                if valid_chord(chord):
                    chords_set.add(chord)
        except Exception as e:
            pass
print(len(chords_set))
print(chords_set)
