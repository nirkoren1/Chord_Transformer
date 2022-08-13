import pychord


if __name__ == '__main__':
    c = pychord.Chord("D#")
    c2 = pychord.Chord("Eb")
    print(c.components())
    print(c2.components())
    print(pychord.find_chords_from_notes(c.components()))
    print(pychord.find_chords_from_notes(c2.components()))
    print(c)