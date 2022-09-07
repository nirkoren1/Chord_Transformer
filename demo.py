from transformer import Transformer
import pickle


with open("word2vec/embeddings.pickle", 'rb') as f:
    embeddings = pickle.load(f)
with open("data_clean/data.pickle", 'rb') as f:
    data = pickle.load(f)

dict_size = len(embeddings)
embeddings_size = 256
padding_size = 256

transformer = Transformer(embeddings_size, 8, dict_size, padding_size)
transformer.load_weights("weights")

if __name__ == '__main__':
    input_ = ["pop", "r&b", "<start>", "G", "C"]
    input_2 = ["pop", "r&b", "rock", "G", "B"]
    transformer.complete(input_, embeddings)
