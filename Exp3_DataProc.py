"""
这个文件中可以添加数据预处理相关函数和类等
如词汇表生成，Word转ID（即词转下标）等
此文件为非必要部分，可以将该文件中的内容拆分进其他部分
"""

import codecs
import numpy as np
import pickle
from tqdm import tqdm
import jieba
import json
from pathlib import Path

def load_dense(path):
    """
    matrix = numpy array of vocab_size rows and embedding_size colomns
    vocab = relations (list or map) between the id and word
    size = embedding dimension
    """
    print("Loading dense word embedding......")
    vocab_size, size = 0, 0
    vocab = {}
    vocab["i2w"], vocab["w2i"] = [""], {"":0}
    count = 1
    with codecs.open(path, "r", "utf-8") as f:
        lines = f.read().split("\n")
        first_line = True
        for line in tqdm(lines):
            if line == "":
                break
            if first_line:
                first_line = False
                vocab_size = int(line.strip().split()[0]) + 1
                size = int(line.rstrip().split()[1])
                matrix = np.zeros(shape=(vocab_size, size), dtype=np.float32)
                continue
            vec = line.strip().split()
            if not vocab["w2i"].__contains__(vec[0]):
                vocab["w2i"][vec[0]] = count
                matrix[count, :] = np.array([float(x) for x in vec[1:]])
                count += 1
    for w, i in vocab["w2i"].items():
        vocab["i2w"].append(w)
    matrix = matrix[:count]
    word_embed = {"matrix": matrix, "vocab_dic": vocab, "embed_dim":size, "vocab_num":count}
    print(f"vocabulary size={word_embed['vocab_num']}, embedding dim={word_embed['embed_dim']}")
    return word_embed 

def add_word(word, word_embed):
    """
        add new word into the word embedding with randomized embedding vector
    """
    if word_embed["vocab_dic"]["w2i"].__contains__(word):
        return 0
    vocab_dic = word_embed["vocab_dic"]
    vocab_dic["i2w"].append(word)
    vocab_dic["w2i"][word] = word_embed["vocab_num"]
    word_embed["vocab_num"] += 1
    return 1

def add_med_word(word_embed):
    print("Loading medical words......")
    with open("./med_word.txt", "r") as f:
        words = f.read().split("\n")
    cnt_new_words = 0
    for word in tqdm(words):
        word = word.strip()
        cnt_new_words += add_word(word, word_embed)
    matrix = word_embed["matrix"]
    embed_dim = word_embed["embed_dim"]
    new_embed = np.random.randn(cnt_new_words, embed_dim)
    word_embed["matrix"] = np.concatenate((matrix, new_embed), axis=0)
    print(f"vocabulary size={word_embed['vocab_num']}")

def add_data_word(word_embed, path, is_test):
    print(f"Loading words in {path}......")
    num = 2 if is_test else 3
    cnt_new_words = 0
    with open(path, "r") as f:
        lines = f.read().split("\n")
        for line in tqdm(lines):
            if line == "":
                break
            words = line.split()
            for i in range(num):
                cnt_new_words += add_word(words[i], word_embed)
    matrix = word_embed["matrix"]
    embed_dim = word_embed["embed_dim"]
    new_embed = np.random.randn(cnt_new_words, embed_dim)
    word_embed["matrix"] = np.concatenate((matrix, new_embed), axis=0)
    print(f"vocabulary size={word_embed['vocab_num']}")

def process_data(word_embed, path, is_test, rel2id):
    print(f"Generating number-formed data from {path}......")
    cnt_new_words = 0
    with open(path, "r") as f:
        out = {"word1":[], "word2":[], "sentence":[], "label":[]}
        lines = f.read().split("\n")
        for line in tqdm(lines):
            if line == "":
                break
            words = line.split()
            if not is_test:
                out["label"].append(rel2id[words[2]])
            sentence = words.pop()
            words += jieba.lcut(sentence)
            for word in words:
                cnt_new_words += add_word(word, word_embed)
            words = list(map(lambda x:word_embed["vocab_dic"]["w2i"][x], words))
            out["word1"].append(words[0])
            out["word2"].append(words[1])
            if is_test:
                out["sentence"].append(words[2:])
            else:
                out["sentence"].append(words[3:])

    matrix = word_embed["matrix"]
    embed_dim = word_embed["embed_dim"]
    new_embed = np.random.randn(cnt_new_words, embed_dim)
    word_embed["matrix"] = np.concatenate((matrix, new_embed), axis=0)

    print(f"vocabulary size={word_embed['vocab_num']}")
    
    with open(path.parent / (path.stem+"_processed.pkl"), "wb") as f:
        print(f"Saving data in format {out.keys()}......")
        pickle.dump(out, f)

if __name__ == '__main__':
    print("Data processing......")
    word_embed = load_dense("./sgns.wiki.word")
    add_med_word(word_embed)
    dir = Path("./data")
    add_data_word(word_embed, dir / "data_train.txt", False)
    add_data_word(word_embed, dir / "data_val.txt", False)
    add_data_word(word_embed, dir / "test_exp3.txt", True)
    with open(dir / "rel2id.json", "r") as f:
        rel2id = json.load(f)[1]
    for word in word_embed["vocab_dic"]["w2i"].keys():
        jieba.add_word(word)
    process_data(word_embed, dir / "data_train.txt", False, rel2id)
    process_data(word_embed, dir / "data_val.txt", False, rel2id)
    process_data(word_embed, dir / "test_exp3.txt", True, rel2id)
    with open("word_embedding.pkl", "wb") as f:
        print(f"Saving word embedding in format {word_embed.keys()}......")
        print(f"Shape of the word embedding array: {word_embed['matrix'].shape}")
        pickle.dump(word_embed, f)
    print("Done.")
