import pickle
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xar = []
yar = []


def write_list(a_list, path):
    # store list in binary file so 'wb' mode
    with open(path, 'wb') as fp:
        pickle.dump(a_list, fp)


def update(y, path):
    yar.append(y)
    xar.append(len(yar))
    write_list(xar, path + "\\xar")
    write_list(yar, path + "\\yar")


def read_list(path):
    # for reading also binary mode is important
    with open(path, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list


def show_fig(path):
    x = read_list(path + "\\xar")
    y = read_list(path + "\\yar")
    fig1 = plt.figure(figsize=(6, 4.5), dpi=100)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.set_ylim(0, 110)
    ax1.set_title('Reward')
    line, = ax1.plot(x, y, 'r', marker='o')
    ax1.set_xlim(-5, len(x) + 5)
    ax1.set_ylim(min(y) - 10, max(y) + 30)
    line.set_data(x, y)
    plt.show()


if __name__ == '__main__':
    show_fig(r"acc_scores")
