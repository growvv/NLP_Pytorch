## 数据预处理
1. 标注化：全部变成小写，筛掉部分标点符号等
2. 变成成对的：
`pairs = [['il est drogue .', 'he s addicted .'], [...], [...] ]`
3. 转成tensor:
`training_pair = tensorsFromPair(random.choice(pairs)`
`input_tensor = training_pair[0]`
`target_tensor = training_pair[1]`
例如：
input_tensor: [[ 118,  214,   51,  202, 2012,    5,    1]] 
target_tensor: [[ 129,   78,  186,  294, 1122,    4,    1]]



## encoder的数据流

![](https://cdn.jsdelivr.net/gh/growvv/image-bed//mac-m1/20210730145119.png)

input_size = input_lang.n_words = 4345
output_size = output_lang.n_words = 2803

1. input: [1,1], 例如input = [[118]]
2. embedded: [1, 1, 256]
3. output: [1, 1, 256], hidden: [1, 1, 256]

## decoder+attn的数据流

![](https://cdn.jsdelivr.net/gh/growvv/image-bed//mac-m1/20210730151525.png)

### 网络

1. embedding； Embedding(2803, 256)
2. attn: Linear(512, 10)
3. attn_combine: Linear(512, 256)
4. gru: GRU(256, 256)
5. out: Linear(256, 2803)

### 数据

1. input: [1, 1], 例如input=[[SOS_Toekn]] = [[0]]
2. embedding: [1, 1, 256]
3. attn_weight: [1, 10]
4. torch.bmm()  三维矩阵乘法 
`
attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
`
[1, 1, 10] x [1, 10, 256] = [1, 1, 256]
5. attn_applied: [1, 1, 256]
6. output: [1, 512]
`output = torch.cat((embedded[0], attn_applied[0]), 1)`
7. output: [1, 256]
`output = self.attn_combine(output).unsqueeze(0)`
8. output, hidden: [1, 1, 256]
`output, hidden = self.gru(output, hidden)`
9. output: [1, 2803]
`output = F.log_softmax(self.out(output[0]), dim=1)`

## 训练
train() 函数
- encoder阶段
input_tensor 一次传一个单词的index进去，得到encoder_outputs 和 最后encoder_hidden

- decode阶段
1. 初始化decoder_hidden = encoder_hidden
2. 初始化decoder_input = torch.tensor([[SOS_token]], device=device), 也就是 [[0]]
`
decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
`

3. 每个 decoder_output 与 targte_tensot[di] 计算loss
`ecoder_output, decoder_hidden, decoder_attention`

## 迭代

trainIters(n_iters) 函数

train(input_tensor, output_tensor) 是每次传入一对数据，若要反复训练，就多调用继承train()

随机选取n_iters对数据，训练n_iters次

## 评估

evaluate(encoder, decoder, sentence)

和train()过程几乎一模一样，因为没有target_tensor，所以decode是非teacher-force方式。每一次预测出一个单词，并添加到output string中，如果预测到EOS token就停止。

我们也会保存decoder的attention outputs，便于后面展示

## 可视化Attention

attention机制有一个良好的特性就是可解释行，因为它是给encoder outputs赋予权重，我们可以知道网络每一步最focus哪里


