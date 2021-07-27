# 基于字符级RNN的名字生成

1. 原始数据集 names.txt

Mary
Annie
Anna
Margaret
Helen
Elsie
...

2. 训练时获取数据

调用 `inp, target = self.get_random_batch()` 获得成对的数据

（1）`file = unidecode.unidecode(open("data/names.txt").read())`

将原始数据拼成一个大字符串 "Mary\nAnnie\nAnna\nMargaret\nHelen\nElsie\n..."

(2) 取file中的部分字符串 `text_str = file[start_idx: end_idx]`

text_str 是长度为chunk_len+1 的字符串

(3) 将字符转成index `text_input[i, :] = self.char_tensor(text_str[:-1])`

例如 text_str = "Mary\nAnnie\nAnna\nMargaret\nHelen\nElsie"

若 index['M'] = 10, index['a'] = 1 ...

text_input[i: ] = [10, 1, ...]

(4) 最终 inp, targte 的 形状为 [batch_size, chunk_len]

例如 
`
inp = [
    [10, 1, 2, ..., 20],
    [20, 4, 2, ..., 11]
]

target = [
    [1, 2, 5, ..., 20],
    [4, 2, 4, ..., 11]
]
`

3. RNN 训练的时候

循环内，每次取一个batch的数据，也就是一列一列的取

例如
`[[10],[20]]`, `[[1],[4]]`, ...