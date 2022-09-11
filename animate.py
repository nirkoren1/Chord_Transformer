import pickle
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xar = [[] for _ in range(5)]
yar = [[] for _ in range(5)]


def write_list(a_list, path):
    # store list in binary file so 'wb' mode
    with open(path, 'wb') as fp:
        pickle.dump(a_list, fp)


def update(y, path, id):
    yar[id].append(y)
    xar[id].append(len(yar[id]))
    write_list(xar[id], path + "\\xar")
    write_list(yar[id], path + "\\yar")


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
    ax1.set_title('Reward')
    line, = ax1.plot(x, y, 'r', marker='o')
    ax1.set_xlim(min(x), len(x) + 5)
    ax1.set_ylim(min(y), max(y))
    line.set_data(x, y)
    plt.show()


if __name__ == '__main__':
    show_fig(r"loss_scores")
    show_fig(r"acc_scores")
