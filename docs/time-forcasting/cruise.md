## 航海大数据

[源代码下载](../../cruise.zip)

1. [BiGRU](#BiGRU)
2. [BiLSTM](#BiLSTM)
3. [BP](#BP)
4. [CNN-GRU](#CG)
5. [CNN-LSTM](#CL)
6. [CNN-LSTM-Attention](#CLA)
7. [GRU](#GRU)
8. [LSTM](#LSTM)
9. [RNN-LSTM](#RL)
10. [RNN-LSTM-Attention](#RLA)
11. [TCN-ABiLSTM](#TABL)
12. [编码解码-LSTM](#EDL)
13. [STA-GRU](#SG)
14. [SW-BiLSTM](#SBL)

#### BiGRU{#BiGRU}

导入设置

```py
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.callbacks import CSVLogger, ReduceLROnPlateau
from keras.optimizers import adam_v2
# import transbigdata as tbd
import warnings

warnings.filterwarnings("ignore")
np.random.seed(120)
tf.random.set_seed(120)
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
```

距离误差函数

```py
def hav(theta):
    s = np.sin(theta / 2)
    return s * s

def get_distance_hav(lat0, lng0, lat1, lng1):
    EARTH_RADIUS = 6371
    lat0 = np.radians(lat0)
    lat1 = np.radians(lat1)
    lng0 = np.radians(lng0)
    lng1 = np.radians(lng1)

    dlng = np.fabs(lng0 - lng1)
    dlat = np.fabs(lat0 - lat1)
    h = hav(dlat) + np.cos(lat0) * np.cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(h))
    return distance
```

用于创造训练集打造标签

```py
def createSequence(data, window=10, maxmin=None):
    train_seq = []
    train_label = []
    m, n = maxmin
    for traj_id in set(data['mmsi']):
        data_temp = data.loc[data.mmsi == traj_id]
        first_lon = data_temp.loc[0, 'lon']
        first_lat = data_temp.loc[0, 'lat']
        end_lon = data_temp.loc[data_temp.shape[0] - 1, 'lon']
        end_lat = data_temp.loc[data_temp.shape[0] - 1, 'lat']

        data_temp = np.array(data_temp.loc[:, ['lon', 'lat', 'sog', 'cog']])
        # 标准化
        data_temp = (data_temp - n) / (m - n)

        for i in range(data_temp.shape[0] - window):
            x = []
            for j in range(i, i + window):
                x.append(list(data_temp[j, :]))
            train_seq.append(x)
            train_label.append(data_temp[i + window, :])

    train_seq = np.array(train_seq, dtype='float64')
    train_label = np.array(train_label, dtype='float64')

    return train_seq, train_label

```

多维反归一化

```py
# 多维反归一化
def FNormalizeMult(y_pre, y_true, max_min):
    [m1, n1, s1, c1], [m2, n2, s2, c2] = max_min
    y_pre[:, 0] = y_pre[:, 0] * (m1 - m2) + m2
    y_pre[:, 1] = y_pre[:, 1] * (n1 - n2) + n2
    y_pre[:, 2] = y_pre[:, 2] * (s1 - s2) + s2
    y_pre[:, 3] = y_pre[:, 3] * (c1 - c2) + c2
    y_true[:, 0] = y_true[:, 0] * (m1 - m2) + m2
    y_true[:, 1] = y_true[:, 1] * (n1 - n2) + n2
    y_true[:, 2] = y_true[:, 2] * (s1 - s2) + s2
    y_true[:, 3] = y_true[:, 3] * (c1 - c2) + c2

    # 计算距离
    y_pre = np.insert(y_pre, y_pre.shape[1],
                      get_distance_hav(y_true[:, 1], y_true[:, 0], y_pre[:, 1], y_pre[:, 0]), axis=1)

    return y_pre, y_true
```



双向GRU层，包含108个单元。 return_sequences=False 参数意味着GRU层的输出将只返回最后一个时间步的输出，而不是返回每个时间步的输出序列。模型使用均方误差（MSE）作为损失函数，优化器为Adam，并且监控准确率（accuracy）作为性能指标。
这个网络结构适用于序列预测任务，例如时间序列分析、自然语言处理等，其中双向GRU层可以捕捉序列数据中的长期依赖关系。

```python title="双向GRU网络"
from keras.layers import GRU
from keras.layers import Bidirectional

def trainModel(train_X, train_Y, test_X, test_Y):
    model = Sequential()
    model.add(Bidirectional(GRU(108, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=False)))
    # model.add(Dropout(0.3))
    model.add(Dense(train_Y.shape[1]))
    model.add(Activation("relu"))
    adam = adam_v2.Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=adam, metrics=['acc'])
    # Save the changes to the log
    log = CSVLogger(f"./log50炼丹1123.csv", separator=",", append=True)
    # Reducing learning rate on a plateau
    reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1, verbose=1,
                               mode='auto', min_delta=0.001, cooldown=0, min_lr=0.001)
    # Model training
    model.fit(train_X, train_Y, epochs=50, batch_size=32, verbose=1, validation_split=0.1,
                  callbacks=[log, reduce])
    # Evaluate with the test set
    loss, acc = model.evaluate(test_X, test_Y, verbose=1)
    print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))
    # Save the model
    model.save(f"./bigru_50_model炼丹1123.h5")
    # Print the neural network structure and count the parameters
    model.summary()
    return model
```

训练过程

```py
# 获取数据
train = pd.read_csv("./train.csv",index_col=0)
test = pd.read_csv("./test.csv",index_col=0)
train.head()

# 计算归一化参数
nor = np.array(train.loc[:, ['lon', 'lat', 'sog', 'cog']])
m = nor.max(axis=0)
n = nor.min(axis=0)
maxmin = [m, n]

# 步长
windows = 10
train_seq, train_label = createSequence(train, windows, maxmin)
test_seq, test_label = createSequence(test, windows, maxmin)
# 训练模型
model = trainModel(train_seq, train_label,test_seq,test_label)
# model = load_model("./bigru_50_model炼丹2.h5")
```

绘制训练结果

```py
logs = pd.read_csv("./log50炼丹1123.csv")

fig, ax = plt.subplots(2,2,figsize=(15,8))
ax[0][0].plot(logs['epoch'],logs['acc'], label='acc')
ax[0][0].set_title('acc')

ax[0][1].plot(logs['epoch'],logs['loss'], label='loss')
ax[0][1].set_title('loss')

ax[1][0].plot(logs['epoch'],logs['val_acc'], label='val_acc')
ax[1][0].set_title('val_acc')

ax[1][1].plot(logs['epoch'],logs['val_loss'], label='val_loss')
ax[1][1].set_title('val_loss')

plt.show()
```

```py
import pandas as pd
import matplotlib.pyplot as plt

logs = pd.read_csv("./log50炼丹1123.csv")

fig, ax = plt.subplots(2, 1, figsize=(6, 6))

# Plot accuracy
ax[0].plot(logs['epoch'], logs['acc'], label='Train Accuracy', color='blue')
ax[0].plot(logs['epoch'], logs['val_acc'], label='Validation Accuracy', color='orange')
ax[0].set_title('Train and Validation Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].legend()

# Plot loss
ax[1].plot(logs['epoch'], logs['loss'], label='Train Loss', color='blue')
ax[1].plot(logs['epoch'], logs['val_loss'], label='Validation Loss', color='orange')
ax[1].set_title('Train and Validation Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend()

plt.tight_layout()
plt.show()

```

```py
test_points_ids = list(set(test['mmsi']))

for ids in test_points_ids[:1]:
    y_pre = []
    test_seq, test_label = createSequence(test.loc[test.mmsi == ids], windows, maxmin)

    y_true = test_label
    for i in range(len(test_seq)):
        y_hat = model.predict(test_seq[i].reshape(1, windows, 4))
        y_pre.append(y_hat[0])
    y_pre = np.array(y_pre, dtype='float64')

    f_y_pre, f_y_true = FNormalizeMult(y_pre, y_true, maxmin)

    print(f"最大值: {max(f_y_pre[:, 4])}\n最小值: {min(f_y_pre[:, 4])}\n均值: {np.mean(f_y_pre[:, 4])}\n"
          f"方差: {np.var(f_y_pre[:, 4])}\n标准差: {np.std(f_y_pre[:, 4])}\n中位数: {np.median(f_y_pre[:, 4])}")

    # 画测试样本数据库
    plt.figure(figsize=(16, 5))
    plt.subplot(121)
    plt.plot(f_y_true[:, 0], f_y_true[:, 1], "ro", markersize=6,label='真实值')
    plt.plot(f_y_pre[:, 0], f_y_pre[:, 1], "bo",markersize=4, label='预测值')
#     bounds = [min(f_y_true[:, 0])-0.02,min(f_y_true[:, 1])-0.01,max(f_y_true[:, 0])+0.02,max(f_y_true[:, 1])+0.01]
#     tbd.plot_map(plt,bounds,zoom = 16,style = 3)
    plt.legend(fontsize=14)
    plt.grid()
    plt.xlabel("经度",fontsize=14)
    plt.ylabel("纬度",fontsize=14)
    plt.title("MMSI:",fontsize=17)

    meanss = np.mean(f_y_pre[:, 4])
    plt.subplot(122)
    plt.bar(range(f_y_pre.shape[0]),f_y_pre[:, 4],label='误差')
    plt.plot([0,f_y_pre.shape[0]],[meanss,meanss],'--r',label="均值")
    plt.title("预测值和真实值的误差",fontsize=17)
    plt.xlabel("船舶轨迹点",fontsize=14)
    plt.ylabel("预测误差(KM)",fontsize=14)
    plt.text(f_y_pre.shape[0]*1.01,meanss*0.96,round(meanss,4),fontsize=14,color='r')
    plt.grid()
    plt.legend(fontsize=14)

    plt.figure(figsize=(16, 6))
    plt.subplot(121)
    plt.plot(f_y_pre[:, 2], "b-", label='预测值')
    plt.plot(f_y_true[:, 2], "r-", label='真实值')
    plt.legend(fontsize=14)
    plt.title("航速预测",fontsize=17)
    plt.xlabel("船舶轨迹点",fontsize=14)
    plt.ylabel("航速/节",fontsize=14)
    plt.grid()

    plt.subplot(122)
    plt.plot(f_y_pre[:, 3], "b-", label='预测值')
    plt.plot(f_y_true[:, 3], "r-", label='真实值')
    plt.legend(fontsize=14)
    plt.title("航向预测",fontsize=17)
    plt.xlabel("船舶轨迹点",fontsize=14)
    plt.ylabel("航向/度",fontsize=14)
    plt.grid()
    
```
循环预测
```py
error_list = []
for ids in test_points_ids[:1]:
    test_seq, test_label = createSequence(test.loc[test.mmsi == ids], windows, maxmin)
    # 要预测的时间
    pre_time = 60
    for start_id in range(test_seq.shape[0]-int(pre_time/2)):
        # 单值预测
        y_pre=[]
        y_true = []
        pre_seq = test_seq[start_id]
        # 循环预测
        for i in range(int(pre_time/2)):
            y_hat = model.predict(pre_seq.reshape(1, windows, 4))
            y_pre.append(y_hat[0])
            y_true.append(test_label[start_id+i])
            # 下一个数组，把预测的值作为预测序列的最后一个值，实现循环预测
            pre_seq = np.insert(pre_seq, pre_seq.shape[0], y_pre[i], axis=0)[1:]

        y_pre = np.array(y_pre, dtype='float64')
        y_true = np.array(y_true, dtype='float64')
        f_y_pre,f_y_true = FNormalizeMult(y_pre,y_true,maxmin)
        error_list.append(list(f_y_pre[:,4]))
```

绘制预测结果

```py
b = np.zeros([len(error_list),len(max(error_list,key = lambda x: len(x)))])
for i,j in enumerate(error_list):
    b[i][0:len(j)] = j

sums = b.sum(axis=0)
maxx = b.max(axis=0)
minx = []
BiGRU_means = []
for col in range(b.shape[1]):
    fzeros = b.shape[0] - list(b[:,col]).count(0.0)
    minx.append(min(list(b[:fzeros,col])))
    BiGRU_means.append(sums[col] / fzeros)

plt.figure(figsize=(12,6))

plt.plot(np.arange(2,2*(b.shape[1])+1,2),BiGRU_means,'-m',label='BiGRU平均误差')
# plt.plot(np.arange(2,2*(b.shape[1])+1,2),minx,'-g',label='最小误差')
# plt.plot(np.arange(2,2*(b.shape[1])+1,2),maxx,'-y',label='最大误差')
plt.xticks(np.arange(2,2*(b.shape[1])+1,2))
plt.yticks(np.arange(0,max(maxx),0.1))
plt.xlabel("时间/分钟",fontsize=14)
plt.ylabel("距离误差/千米",fontsize=14)
plt.legend(fontsize=14)
plt.grid()
plt.title("整条轨迹上随时间变化的预测距离误差",fontsize=17)
plt.savefig('六种模型随时间平均误差对比.png')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(f_y_true[:, 1], f_y_pre[:, 1])
print(f"平均绝对误差（MAE）: {mae}")
mse = mean_squared_error(f_y_true[:, 1], f_y_pre[:, 1])
rmse = np.sqrt(mse)
print(f"均方根误差（RMSE）: {rmse}")
r2 = r2_score(f_y_true[:, 1], f_y_pre[:, 1])
print(f"R2分数（R2_score）: {r2}")

```

```py
plt.figure(figsize=(16,8))
plt.subplot(121)
plt.plot(f_y_pre[:, 0], f_y_pre[:, 1], "-b", label='预测值')
plt.plot(f_y_true[:, 0], f_y_true[:, 1], "-r", label='真实值')
# plt.plot(true_lables[:start_id, 0], true_lables[:start_id, 1], "o",color='#eef200', label='历史位置')
# bounds = [min(f_y_true[:, 0])-0.01,min(f_y_true[:, 1])-0.01,max(f_y_true[:, 0])+0.01,max(f_y_true[:, 1])+0.01]
# tbd.plot_map(plt,bounds,zoom = 16,style = 3)
plt.legend(fontsize=15)
plt.title(f'预测步数量={maxStep},开始位置={start_id}',fontsize=17)
plt.title(f'真实轨迹与预测轨迹',fontsize=17)
plt.xlabel("经度",fontsize=15)
plt.ylabel("纬度",fontsize=15)
plt.grid()
```

#### BiLSTM{#BiLSTM}

- 添加了一个双向长短期记忆网络（Bidirectional LSTM），其中包含108个LSTM单元。
- `input_shape`参数设置为`(train_X.shape[1], train_X.shape[2])`，这意味着网络的输入数据形状由`train_X`的第二和第三维度决定。
- 使用Adam优化器（`adam_v2.Adam`），学习率设置为0.01。
- 模型的损失函数设置为均方误差（`mse`），评估指标设置为准确率（`'acc'`）
- 使用`ReduceLROnPlateau`回调函数来在验证集准确率不再提升时降低学习率。

```py title="BiLSTM"
from keras.layers import Bidirectional

def trainModel(train_X, train_Y, test_X, test_Y):
    model = Sequential()
    model.add(Bidirectional(LSTM(108, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=False)))
    # model.add(Dropout(0.3))
    model.add(Dense(train_Y.shape[1]))
    model.add(Activation("relu"))
    adam = adam_v2.Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=adam, metrics=['acc'])
    # Save the changes to the log
    log = CSVLogger(f"./log50炼丹1.csv", separator=",", append=True)
    # Reducing learning rate on a plateau
    reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1, verbose=1,
                               mode='auto', min_delta=0.001, cooldown=0, min_lr=0.001)
    # Model training
    model.fit(train_X, train_Y, epochs=50, batch_size=32, verbose=1, validation_split=0.1,
                  callbacks=[log, reduce])
    # Evaluate with the test set
    loss, acc = model.evaluate(test_X, test_Y, verbose=1)
    print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))
    # Save the model
    model.save(f"./biLSTM_50_model炼丹1.h5")
    # Print the neural network structure and count the parameters
    model.summary()
    return model
```

训练过程、绘制结果、模型预测、预测结果参上[BiGRU](#BiGRU)

#### BP{#BP}

1. **模型初始化**：
   - 使用`Sequential()`创建了一个顺序模型。
2. **Flatten层**：
   - 添加了一个`Flatten`层，用于将输入数据展平成一维数组，以适应后续的全连接层。
   - `input_shape`参数设置为`(train_X.shape[1], train_X.shape[2])`，这意味着网络的输入数据形状由`train_X`的第二和第三维度决定。
3. **全连接层（Dense）**：
   - 第一个全连接层有64个单元，使用ReLU激活函数。
   - 第二个全连接层有32个单元，同样使用ReLU激活函数。
   - 输出层的单元数等于`train_Y.shape[1]`，即目标变量的维度。
4. **优化器和损失函数**：
   - 使用Adam优化器，学习率设置为0.01。
   - 模型的损失函数设置为均方误差（`mse`），评估指标设置为准确率（`'acc'`）。
5. **日志记录器（CSVLogger）**：
   - 使用`CSVLogger`来记录训练过程中的日志信息，保存到`./bp_50炼丹.csv`文件中。
6. **学习率调整（ReduceLROnPlateau）**：
   - 使用`ReduceLROnPlateau`回调函数来在验证集准确率不再提升时降低学习率。
7. **模型训练（model.fit）**：
   - 使用`model.fit`方法训练模型，设置50个训练周期（epochs），每批32个样本，并且使用10%的数据作为验证集。
   - 训练过程中会使用日志记录器和学习率调整器。
8. **模型评估**：
   - 使用`model.evaluate`方法在测试集上评估模型，输出损失和准确率。
9. **模型保存**：
   - 将训练好的模型保存到`./BP_50_model炼丹.h5`文件中。
10. **模型结构和参数统计**：
    - 使用`model.summary()`打印模型的结构和参数数量。

```py title="BP神经网络"
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ReduceLROnPlateau

def trainModel(train_X, train_Y, test_X, test_Y):
    model = Sequential()
    model.add(Flatten(input_shape=(train_X.shape[1], train_X.shape[2])))  # 将数据展平以适应输入层
    model.add(Dense(64, activation='relu'))  # 输入层
    model.add(Dense(32, activation='relu'))  # 隐藏层
    model.add(Dense(train_Y.shape[1]))  # 输出层

    adam = Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=adam, metrics=['acc'])
    
    log = CSVLogger(f"./bp_50炼丹.csv", separator=",", append=True)
    reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1, verbose=1, mode='auto', min_delta=0.001, cooldown=0, min_lr=0.001)
    model.fit(train_X, train_Y, epochs=50, batch_size=32, verbose=1, validation_split=0.1, callbacks=[log, reduce])
    
    loss, acc = model.evaluate(test_X, test_Y, verbose=1)
    print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))
    
    model.save(f"./BP_50_model炼丹.h5")
    model.summary()
    return model
```

其余过程参上[BiGRU](#BiGRU)

#### CNN-GRU{#CG}

- 两个卷积层有64个滤波器，核大小为2，使用ReLU激活函数，并且设置`padding='same'`以保持输出尺寸与输入相同
- 使用`MaxPooling1D`层，池化窗口大小为2，用于降低特征维度。

- 使用`Reshape`层将全连接层的输出重塑为`(1, dense1.shape[1])`的形状，以适应GRU层的输入要求。
- 第一个`GRU`层有108个单元，返回序列。第二个`GRU`层与第一个配置相同。第三个`GRU`层有108个单元，不返回序列。

```py title="CNN-GRU"
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, GRU, Reshape, Add, Attention
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ReduceLROnPlateau
import numpy as np

def trainModel(train_X, train_Y, test_X, test_Y):
    input_layer = Input(shape=(train_X.shape[1], train_X.shape[2]))
    conv1 = Conv1D(filters=64, kernel_size=2, activation='relu', padding='same')(input_layer)
    conv2 = Conv1D(filters=64, kernel_size=2, activation='relu', padding='same')(conv1)
    max_pooling = MaxPooling1D(pool_size=2)(conv2)
    flatten = Flatten()(max_pooling)
    dense1 = Dense(100, activation='relu')(flatten)
    
    reshaped = Reshape((1, dense1.shape[1]))(dense1)


    gru1 = GRU(108, return_sequences=True)(reshaped)
#     gru1_dropout = Dropout(0.1)(gru1)  # Adding Dropout after the first GRU layer
    gru2 = GRU(108, return_sequences=True)(gru1)
    gru3 = GRU(108, return_sequences=False)(gru2)
    dense2 = Dense(50, activation='relu')(gru3)
    output_layer = Dense(train_Y.shape[1], activation='relu')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    # 设置优化器
    adam = Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=adam, metrics=['acc'])

    # 保存训练过程中的日志
    log = CSVLogger(f"./CNN_GRU_log.csv", separator=",", append=True)

    # 设置自适应学习率调整策略
    reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1, verbose=1, mode='auto', min_delta=0.001, cooldown=0, min_lr=0.001)

    # 开始模型训练
    model.fit(train_X, train_Y, epochs=50, batch_size=32, verbose=1, validation_split=0.1, callbacks=[log, reduce])

    # 在测试集上评估模型性能
    loss, acc = model.evaluate(test_X, test_Y, verbose=1)
    print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))

    # 保存模型
    model.save(f"./CNN_GRU_model.h5")

    # 打印模型结构和参数统计
    model.summary()

    return model
```

其余过程参上[BiGRU](#BiGRU)

#### CNN-LSTM{#CL}

- 使用`Input`层定义了模型的输入形状，即`(train_X.shape[1], train_X.shape[2])`。
- 第一个`Conv1D`层有32个滤波器，核大小为2，使用ReLU激活函数，并且设置`padding='same'`以保持输出尺寸与输入相同。
- 使用`MaxPooling1D`层，池化窗口大小为2，用于降低特征维度。
- 使用`Dropout`层，丢弃率为0.2，用于减少过拟合。
- 使用`Reshape`层将全连接层的输出重塑为`(1, dense1.shape[1])`的形状，以适应LSTM层的输入要求。

- 使用`LSTM`层，有100个单元，不返回序列。

- 使用`Model`类构建模型，输入层为`input_layer`，输出层为`output_layer`。

```py
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, GRU, Reshape, Add, Attention,Dropout
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ReduceLROnPlateau
import numpy as np

def trainModel(train_X, train_Y, test_X, test_Y):
    input_layer = Input(shape=(train_X.shape[1], train_X.shape[2]))
    conv1 = Conv1D(filters=32, kernel_size=2, activation='relu', padding='same')(input_layer)
    max_pooling = MaxPooling1D(pool_size=2)(conv1)
    dropout = Dropout(0.2)(max_pooling)
    conv2 = Conv1D(filters=32, kernel_size=2, activation='relu', padding='same')(dropout)
#     max_pooling = MaxPooling1D(pool_size=2)(conv2)
    flatten = Flatten()(max_pooling)
#     residual1 = Add()([conv1, conv2])  # Adding the residual connection
    dense1 = Dense(100, activation='relu')(flatten)
#     dense2 = Dense(32, activation='relu')(dense1)
#     dense3 = Dense(16, activation='relu')(dense2)
    reshaped = Reshape((1, dense1.shape[1]))(dense1)

    # Adding Attention layer after all the previous layers
#     attention = Attention()([reshaped, reshaped])
#     attended_input = Add()([reshaped, attention])
    lstm1 = LSTM(100, return_sequences=False)(reshaped)
#     lstm2 = LSTM(108, return_sequences=False)(lstm1)
#     lstm = GRU(108, return_sequences=True)(attended_input)
#     lstm_dropout = Dropout(0.2)(lstm)  # Adding Dropout after the first GRU layer
#     gru2 = GRU(108, return_sequences=False)(gru1)
    output_layer = Dense(train_Y.shape[1], activation='relu')(lstm1)

    model = Model(inputs=input_layer, outputs=output_layer)

    # 设置优化器
    adam = Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=adam, metrics=['acc'])

    # 保存训练过程中的日志
    log = CSVLogger(f"./CNN_LSTM_log50_2.csv", separator=",", append=True)

    # 设置自适应学习率调整策略
    reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1, verbose=1, mode='auto', min_delta=0.001, cooldown=0, min_lr=0.001)

    # 开始模型训练
    model.fit(train_X, train_Y, epochs=50, batch_size=32, verbose=1, validation_split=0.1, callbacks=[log, reduce])

    # 在测试集上评估模型性能
    loss, acc = model.evaluate(test_X, test_Y, verbose=1)
    print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))

    # 保存模型
    model.save(f"./CNN_LSTM_model_炼丹2.h5")

    # 打印模型结构和参数统计
    model.summary()

    return model

```

其余过程参上[BiGRU](#BiGRU)

#### CNN-LSTM-Att{#CLA}

- 使用`Attention`层对最后一个LSTM层的输出进行注意力加权。

- 使用`Add`层将注意力加权后的输出与原始LSTM输出相加，得到增强的输入。

```py title="CNN-LSTM-Attention"
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, GRU, Reshape, Add, Attention, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ReduceLROnPlateau
import numpy as np
from tcn import TCN 
def trainModel(train_X, train_Y, test_X, test_Y):
    input_layer = Input(shape=(train_X.shape[1], train_X.shape[2]))
    conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_layer)
#     conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(conv1)
    lstm1 = LSTM(32, return_sequences=True,activation='tanh')(conv1)
    lstm2 = LSTM(32, return_sequences=True,activation='tanh')(lstm1)
    lstm3 = LSTM(32, return_sequences=False,activation='tanh')(lstm2)

    attention = Attention()([lstm3, lstm3])
    attended_input = Add()([lstm3, attention])

#     biLSTM = Bidirectional(LSTM(108, return_sequences=False, activation='tanh'))(attended_input)
#     gru1_dropout = Dropout(0.1)(gru1)  # Adding Dropout after the first GRU layer
#     gru1 = GRU(489, return_sequences=True,activation='tanh')(attended_input)
#     gru2 = GRU(30, return_sequences=False,activation='tanh')(gru1)
#     dense2 = Dense(dense_neurons2, activation='relu')(bigru1)
    output_layer = Dense(train_Y.shape[1], activation='relu')(attended_input)
    model = Model(inputs=input_layer, outputs=output_layer)
    adam = Adam(learning_rate=0.01)

    model.compile(loss='mse', optimizer=adam, metrics=['acc'])

    # 保存训练过程中的日志
    log = CSVLogger(f"./CNN-LSTM_A_log.csv", separator=",", append=True)

    # 设置自适应学习率调整策略
    reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1, verbose=1, mode='auto', min_delta=0.001, cooldown=0, min_lr=0.001)

    # 开始模型训练
    model.fit(train_X, train_Y, epochs=50, batch_size=32, verbose=1, validation_split=0.1, callbacks=[log, reduce])


    # 在测试集上评估模型性能
    loss, acc = model.evaluate(test_X, test_Y, verbose=1)
    print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))

    # 保存模型
    model.save(f"./CNN-LSTM-_A_model.h5")

    # 打印模型结构和参数统计
    model.summary()

    return model
```

其余过程参上[BiGRU](#BiGRU)

#### GRU{#GRU}

- 使用`Sequential()`创建了一个顺序模型。

  **GRU层**：

- 添加了一个GRU层，包含108个单元。
- `input_shape`参数设置为`(train_X.shape[1], train_X.shape[2])`，这意味着网络的输入数据形状由`train_X`的第二和第三维度决定。
- `return_sequences`参数设置为`False`，表示这个GRU层的输出不会返回序列，而是返回最后一个时间步的输出。

```py title="GRU"
from keras.layers import GRU

def trainModel(train_X, train_Y, test_X, test_Y):
    model = Sequential()
    model.add(GRU(108, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=False))
    # model.add(Dropout(0.3))
    model.add(Dense(train_Y.shape[1]))
    model.add(Activation("relu"))
    adam = adam_v2.Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=adam, metrics=['acc'])
    # 保存训练过程中损失函数和精确度的变化
    log = CSVLogger(f"./log50炼丹3.csv", separator=",", append=True)
    # 用来自动降低学习率
    reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1, verbose=1,
                               mode='auto', min_delta=0.001, cooldown=0, min_lr=0.001)
	# 模型训练
    model.fit(train_X, train_Y, epochs=50, batch_size=32, verbose=1, validation_split=0.1,
                  callbacks=[log, reduce])
    # 用测试集评估
    loss, acc = model.evaluate(test_X, test_Y, verbose=1)
    print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))
    # 保存模型
    model.save(f"./GRU_50_model炼丹3_.h5")
    # 打印神经网络结构，统计参数数目
    model.summary()
    return model
```

其余过程参上[BiGRU](#BiGRU)

#### LSTM{#LSTM}

- `equential()`：创建一个顺序模型。

- `LSTM(108, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=False)`：添加一个LSTM层，有108个单元，输入形状由`train_X`的第二和第三维度确定，`return_sequences=False`表示LSTM层的输出不会返回序列。
- `Dense(train_Y.shape[1])`：添加一个全连接层，神经元的数量由`train_Y`的第二维度确定。
- `Activation("relu")`：激活函数设置为ReLU.

```py title="LSTM"
def trainModel(train_X, train_Y, test_X, test_Y):
    model = Sequential()
    model.add(LSTM(108, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=False))
    # model.add(Dropout(0.3))
    model.add(Dense(train_Y.shape[1]))
    model.add(Activation("relu"))
    adam = adam_v2.Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=adam, metrics=['acc'])
    log = CSVLogger(f"./log30炼丹1.csv", separator=",", append=True)
    reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1, verbose=1,
                               mode='auto', min_delta=0.001, cooldown=0, min_lr=0.001)

    model.fit(train_X, train_Y, epochs=30, batch_size=32, verbose=1, validation_data=(val_X, val_Y), callbacks=[log, reduce])
    loss, acc = model.evaluate(test_X, test_Y, verbose=1)
    print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))
    model.save(f"./lstm_30_model炼丹1.h5")
    # 打印神经网络结构，统计参数数目
    model.summary()
    return model
```

其余过程参上[BiGRU](#BiGRU)

#### RNN-LSTM{#RL}

- `Sequential()`：创建一个顺序模型。
- `SimpleRNN(60, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True)`：添加一个SimpleRNN层，有60个单元，输入形状由`train_X`的第二和第三维度确定，`return_sequences=True`表示输出序列。
- `LSTM(60, return_sequences=False)`：添加一个LSTM层，有60个单元，`return_sequences=False`表示输出不是序列。
- `Dense(train_Y.shape[1])`：添加一个全连接层，神经元数量由`train_Y`的第二维度确定。
- `Activation("relu")`：激活函数设置为ReLU。

```py title="RNN-LSTM"
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Activation,SimpleRNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau

def trainModel(train_X, train_Y, test_X, test_Y):
    model = Sequential()
    model.add(SimpleRNN(60, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(LSTM(60, return_sequences=False))
    # model.add(Dropout(0.3))
    model.add(Dense(train_Y.shape[1]))
    model.add(Activation("relu"))
    adam = Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=adam, metrics=['acc'])
    # 保存训练过程中损失函数和精确度的变化
    log = CSVLogger(f"./RNN_LSTM_log50_2.csv", separator=",", append=True)
    # 用来自动降低学习率
    reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1, verbose=1,
                               mode='auto', min_delta=0.001, cooldown=0, min_lr=0.001)
    # 模型训练
    model.fit(train_X, train_Y, epochs=50, batch_size=32, verbose=1, validation_split=0.1,
                  callbacks=[log, reduce])
    # 用测试集评估
    loss, acc = model.evaluate(test_X, test_Y, verbose=1)
    print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))
    # 保存模型
    model.save(f"./RNN_LSTM_model2.h5")
    # 打印神经网络结构，统计参数数目
    model.summary()
    return model

```

其余过程参上[BiGRU](#BiGRU)

#### RNN-LSTM-Att{#RLA}

- 输入数据首先通过一个自注意力机制（`Attention`层），这个层会计算输入数据的注意力权重，然后将这些权重应用到输入数据上。
- 自注意力层的输出被送入一个LSTM层，这个LSTM层有64个神经元，不返回序列（`return_sequences=False`），使用双曲正切激活函数（`'tanh'`）
- 为了减少过拟合，LSTM层的输出通过一个Dropout层，丢弃率为0.2。
- Dropout层的输出连接到一个全连接层，这个层有64个神经元，同样使用双曲正切激活函数。
- 最终，全连接层的输出被送入输出层，这个层的神经元数量由`train_Y`的第二维度决定，也使用双曲正切激活函数。
- 模型使用Adam优化器，学习率为0.01，损失函数为均方误差（`'mse'`），并且监控准确率（`'acc'`）

```py title="RNN-LSTM-Attention"
#模型构建
import tensorflow as tf
from tensorflow.keras import layers,Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Conv1D, Dropout, Attention, Concatenate,LayerNormalization
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * np.exp(-0.1)
def trainModel(train_X, train_Y, test_X, test_Y):
    model = Sequential()
    input_layer = layers.Input(shape=(train_X.shape[1], train_X.shape[2]))
    #神经元（卷积核）20个，卷积核大小6，膨胀大小为2的次方
    #t=TCN(return_sequences=True,nb_filters=64,kernel_size=5,dilations=[2 ** i for i in range(9)])(input_layer)
    
    # Step 2: Attention Mechanism
    self_attention_input = input_layer

    # 注意力层计算
    atten = Attention()([self_attention_input, self_attention_input])
    lstm = LSTM(64, return_sequences=False,activation='tanh')(atten)
    drop = Dropout(0.2)(lstm)
    dense1=Dense(64,activation='tanh')(drop)
    output_layer=Dense(train_Y.shape[1],activation='tanh')(dense1)
    model = Model(inputs=input_layer,outputs=output_layer)
    adam = Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=adam, metrics=['acc'])
    log = CSVLogger(f"./Atten-LSTM_log.csv", separator=",", append=True)
    # 用来自动降低学习率
    reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1, verbose=1,
                               mode='auto', min_delta=0.001, cooldown=0, min_lr=0.001)
    lrs = LearningRateScheduler(scheduler)
    # 开始模型训练
    model.fit(train_X, train_Y, epochs=50, batch_size=64, verbose=1, validation_split=0.2, callbacks=[log])
    loss, acc = model.evaluate(test_X, test_Y, verbose=1)
    print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))
    model.save(f"D:/jupyter/LSTM_ship/Liuxin/Atten-LSTM/Atten-LSTM_model.h5")
    # 打印神经网络结构，统计参数数目
    model.summary()
    return model
```

其余过程参上[BiGRU](#BiGRU)

#### TCN-ABiLSTM{#TABL}

1. **全局注意力机制**：
   - 定义了一个自定义的`GlobalAttention`层，该层计算输入的query和value之间的点积，然后应用softmax函数获取权重，最后计算加权的value。
   - 输入层的数据通过三个全连接层（Dense）分别转换为query、key和value。
   - 使用`GlobalAttention`层处理query和value，得到注意力机制的输出。
   - 将注意力输出与原始输入层进行拼接。
2. **双向长短期记忆网络（Bi-LSTM）**：拼接后的输出被送入一个双向LSTM层，该层有64个神经元，不返回序列（`return_sequences=False`），使用双曲正切激活函数（`'tanh'`）。

```py title="TCN-ABiLSTM"
#模型构建
import tensorflow as tf
from tensorflow.keras import layers,Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Conv1D, Dropout, Attention, Concatenate,LayerNormalization
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * np.exp(-0.1)
def trainModel(train_X, train_Y, test_X, test_Y):
    model = Sequential()
    
    # Step 1: TCN
    input_layer = layers.Input(shape=(train_X.shape[1], train_X.shape[2]))
    #神经元（卷积核）20个，卷积核大小6，膨胀大小为2的次方
    #t=TCN(return_sequences=True,nb_filters=64,kernel_size=5,dilations=[2 ** i for i in range(9)])(input_layer)
    
    # Step 2: Attention Mechanism
    class GlobalAttention(Layer):
        def __init__(self, **kwargs):
            super(GlobalAttention, self).__init__(**kwargs)
    
        def call(self, inputs):
            query, value = inputs
        
            # Compute the dot product between query and key
            score = tf.matmul(query, value, transpose_b=True)
            weights = tf.nn.softmax(score, axis=-1)
        
            # Compute the weighted sum of the values
            attention_output = tf.matmul(weights, value)
        
            return attention_output


    # Apply Dense layers to prepare query, key, and value
    query = Dense(64)(input_layer)
    key = Dense(64)(input_layer)
    value = Dense(64)(input_layer)

    # Apply the global attention mechanism
    attention_output = GlobalAttention()([query, value])

    # Concatenate the attention output with the TCN output
    atten = Concatenate()([input_layer, attention_output])

    #att = Attention()([t, t])
    #atten = Concatenate()([t, att])
    
    #Bi-LSTM层
    bi_lstm = Bidirectional(LSTM(64, return_sequences=False,activation='tanh'))(atten)
    drop = Dropout(0.2)(bi_lstm)
    
    # Step 3: 全连接层
    dense1=Dense(50,activation='tanh')(drop)
    output_layer=Dense(train_Y.shape[1],activation='tanh')(dense1)
    model = Model(inputs=input_layer,outputs=output_layer)
    adam = Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=adam, metrics=['acc'])
    # 保存训练过程中损失函数和精确度的变化
    log = CSVLogger(f"./log.csv", separator=",", append=True)
    # 用来自动降低学习率
    reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1, verbose=1,
                               mode='auto', min_delta=0.001, cooldown=0, min_lr=0.001)
    lrs = LearningRateScheduler(scheduler)
    # 模型训练
    model.fit(train_X, train_Y, epochs=50, batch_size=32, verbose=1, validation_split=0.2,
                  callbacks=[log, reduce,lrs])
    # 用测试集评估
    loss, acc = model.evaluate(test_X, test_Y, verbose=1)
    print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))
    # 保存模型
    model.save(f"./model.h5")
    # 打印神经网络结构，统计参数数目
    model.summary()
    return model
```

其余过程参上[BiGRU](#BiGRU)

#### 编码解码-LSTM{#EDL}

**编码器（Encoder）部分：**

1. **输入层**：`encoder_inputs` 是编码器的输入，其形状由 `train_X` 的第二和第三维度决定，即 `shape=(train_X.shape[1], train_X.shape[2])`。
2. **LSTM层**：`encoder_lstm` 是一个 LSTM 层，有 100 个神经元，并且返回状态。这个层接收 `encoder_inputs` 作为输入，并输出序列、隐藏状态 `state_h` 和细胞状态 `state_c`。这两个状态将被用作解码器的初始状态。
3. **状态输出**：`encoder_states` 是一个包含隐藏状态和细胞状态的列表，将被传递给解码器。

**解码器（Decoder）部分：**

1. **输入层**：`decoder_inputs` 是解码器的输入，其形状由 `train_Y` 的第二和第三维度决定，即 `shape=(train_Y.shape[1], train_Y.shape[2])`。
2. **LSTM层**：`decoder_lstm` 是一个 LSTM 层，有 100 个神经元，返回序列和状态。这个层接收 `decoder_inputs` 作为输入，并使用编码器的最终状态作为初始状态。它输出序列。
3. **全连接层**：`decoder_dense` 是一个全连接层（Dense），有 `train_Y.shape[2]` 个神经元，使用 ReLU 激活函数。这个层接收 LSTM 层的输出，并输出最终的预测。

**模型构建：**

- 使用 `Model` 类构建模型，输入是编码器和解码器的输入，输出是解码器的输出。

```py title = "encoder-decoder-LSTM"
def trainModel(train_X, train_Y, test_X, test_Y):
    # Encoder部分
    encoder_inputs = Input(shape=(train_X.shape[1], train_X.shape[2]))
    encoder_lstm = LSTM(100, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder部分
    decoder_inputs = Input(shape=(train_Y.shape[1], train_Y.shape[2]))  # 目标序列的形状
    decoder_lstm = LSTM(100, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(train_Y.shape[2], activation='relu')
    decoder_outputs = decoder_dense(decoder_outputs)

    # 构建模型
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # 编译模型
    model.compile(optimizer='adam', loss='mse', metrics=['acc'])

    # 记录日志
    log = CSVLogger(f"./log_encoder_decoder_lstm.csv", separator=",", append=True)

    # 训练模型
    model.fit([train_X, train_Y], train_Y, epochs=200, batch_size=64, verbose=1, validation_split=0.1, callbacks=[log])

    # 模型评估
    loss, acc = model.evaluate([test_X, test_Y], test_Y, verbose=1)
    print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))

    # 保存模型
    model.save(f"./encoder_decoder_lstm_model.h5")

    # 打印模型结构
    model.summary()

    return model
```

其余过程参上[BiGRU](#BiGRU)

#### STA-GRU{#SG}

**自定义注意力模块（AttentionBlock）：**

1. **初始化**：`AttentionBlock` 类继承自 `Layer`，初始化时接收输入维度 `input_dim`，并定义一个 `Dense` 层用于计算注意力权重，激活函数为 `softmax`。
2. **前向传播**：`call` 方法接收输入 `inputs`，使用 `Dense` 层计算注意力权重，然后使用 `Multiply` 层将注意力权重应用到输入上，输出加权后的结果。
3. **配置获取**：`get_config` 方法用于序列化层的配置，以便后续可以重建相同的层。

**模型构建：**

1. **注意力模块**：使用自定义的 `AttentionBlock` 作为模型的第一层，输入维度为 `train_X.shape[2]`，输入形状为 `(train_X.shape[1], train_X.shape[2])`。
2. **GRU层**：添加一个 GRU 层，有100个单元，不返回序列（`return_sequences=False`）。
3. **Dropout层**：添加一个 Dropout 层，丢弃率为0.5，用于防止过拟合。
4. **输出层**：添加一个全连接层（Dense），输出维度与标签 `train_Y` 的第二维度相同。
5. **激活函数**：在输出层后添加 ReLU 激活函数。

```py title="自定义注意力层的GRU"
class AttentionBlock(Layer):
    def __init__(self, input_dim, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.input_dim = input_dim
        # 定义 Dense 层用于计算注意力权重
        self.attention_dense = Dense(input_dim, activation='softmax')

    def call(self, inputs):
        # 计算注意力权重，不进行转置
        attention_weights = self.attention_dense(inputs)
        # 应用注意力权重到输入上，形状必须一致
        output_attention = Multiply()([inputs, attention_weights])
        return output_attention

    def get_config(self):
        config = super(AttentionBlock, self).get_config()
        config.update({'input_dim': self.input_dim})
        return config

def trainModel(train_X, train_Y, test_X, test_Y):
    model = Sequential()

   # 使用自定义的 AttentionBlock 替代 Lambda 层
    model.add(AttentionBlock(input_dim=train_X.shape[2], input_shape=(train_X.shape[1], train_X.shape[2])))
    
    # GRU层
    model.add(GRU(100, return_sequences=False))  # 使用GRU单元
    model.add(Dropout(0.5))  # Dropout层，防止过拟合
    
    # 输出层
    model.add(Dense(train_Y.shape[1]))  # 输出层，输出维度与标签维度相同
    model.add(Activation("relu"))  # 使用ReLU激活函数

    # 使用Adam优化器
    adam = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=adam, metrics=['acc'])  # 使用均方误差（MSE）作为损失函数

    # 记录训练日志
    log = CSVLogger(f"./log_sta_gru.csv", separator=",", append=True)

    # 开始训练模型
    model.fit(train_X, train_Y, epochs=200, batch_size=64, verbose=1, validation_split=0.1, callbacks=[log])

    # 模型评估
    loss, acc = model.evaluate(test_X, test_Y, verbose=1)
    print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))

    # 保存模型
    model.save(f"./sta_gru_model.h5")

    # 打印模型结构
    model.summary()

    return model
```

#### SW-BiLSTM{#SBL}

包含卷积层、最大池化层、Dropout层、双向长短期记忆网络（Bi-LSTM）和全连接输出层的神经网络模型。

1. **卷积层**：模型的第一层是一维卷积层（`Conv1D`），有64个过滤器（filters），核大小为3，使用“same”填充以保持输出尺寸与输入相同，激活函数为ReLU。
2. **最大池化层**：接着是一个最大池化层（`MaxPooling1D`），池化窗口大小为2，用于降低特征维度并提取重要特征。
3. **Dropout层**：添加一个Dropout层，丢弃率为0.5，用于防止过拟合。
4. **双向LSTM层**：使用 `Bidirectional` 包装一个 LSTM 层，LSTM层有100个单元，不返回序列（`return_sequences=False`），用于处理序列数据。
5. **输出层**：添加一个全连接层（Dense），输出维度与标签 `train_Y` 的维度相同。
6. **激活函数**：在输出层后添加ReLU激活函数。

```py title="SW-BiLSTM"
def trainModel(train_X, train_Y, test_X, test_Y):
    model = Sequential()
    
     # 引入卷积层作为滑动窗口特征提取器
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(MaxPooling1D(pool_size=2))  # 最大池化层减少特征维度
    model.add(Dropout(0.5))  # Dropout层防止过拟合
    
    # 使用双向LSTM（Bi-LSTM）
    model.add(Bidirectional(LSTM(100, return_sequences=False), input_shape=(train_X.shape[1], train_X.shape[2])))
    
    #输出层
    model.add(Dense(train_Y.shape[1]))  # 输出层，输出维度与标签维度相同
    model.add(Activation("relu"))  # 使用ReLU激活函数
    
    # 使用优化后的Adam优化器
    adam = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=adam, metrics=['acc'])  # 使用均方误差（MSE）作为损失函数
    
    # 记录训练日志
    log = CSVLogger(f"./log_sw_bi_lstm.csv", separator=",", append=True)
  
    # 开始训练模型，相同参数
    model.fit(train_X, train_Y, epochs=200, batch_size=64, verbose=1, validation_split=0.1, callbacks=[log])
    
    # 模型评估
    loss, acc = model.evaluate(test_X, test_Y, verbose=1)
    print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))
    
     # 保存模型
    model.save(f"./sw_bi_lstm_model.h5")
    
    # 打印神经网络结构，统计参数数目
    model.summary()
    
    return model
```

