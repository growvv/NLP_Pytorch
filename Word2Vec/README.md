## CBOW的原理

### 网络图
![](https://img2020.cnblogs.com/blog/1365470/202107/1365470-20210720094243319-1690538018.png)

### 训练时

假如 nn.Embedding(10, 3)

则V=10, N=3，V表示所有词汇的个数，也就是one-hot时的长度，N表示embedding的长度

也就是

1. $[1 \times V]_{one-hot} \times [V \times N]_{W} = [1 \times N]_{embedding}$
2. 实际上这里 $[1 \times N]_{embedding}$ 前 $C$ 个相加得到的
3. $[1 \times N]_{embedding} \times [N \times V]_{W'} = [1 \times V]_{one-hot'}$
4. 将得到的 $[1 \times V]_{one-hot'}$ 与 target 计算误差