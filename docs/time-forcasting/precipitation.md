

## 降雨量预测

[源代码链接](../../precipitation.rar)

1. [刘新宇](#lxy)
1. [徐华](#XH)

	1. [LSTM](#LSTM)
	1. [CEEMDAN(改进经验模态分解)](#CEEMDAN)

#### 刘新宇{#lxy}

程序入口

```py title='run'

#RNN时间序列
def main():
    #读取所需参数
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    #读取数据
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
    #创建RNN模型
    model = Model()
    mymodel = model.build_model(configs)
    
    plot_model(mymodel, to_file='model.png',show_shapes=True)
    
    #加载训练数据
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    print (x.shape)
    print (y.shape)
    
	#训练模型
    model.train(
		x,
		y,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
		save_dir = configs['model']['save_dir']
	)
	
   #测试结果
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    
    #展示测试效果
    predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'],debug=False)
    print (np.array(predictions).shape)

    plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
    
if __name__ == '__main__':
    main()
```

数据预处理模块

```py
import math
import numpy as np
import pandas as pd

class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split, cols):
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test  = dataframe.get(cols).values[i_split:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x,y

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
```



**模型**

这段代码定义了一个名为 `Model` 的类，用于构建和加载基于长短期记忆网络（LSTM）的神经网络模型。

构建模型的步骤：

1. **初始化计时器**：使用 `Timer` 开始计时。

2. **遍历配置**：遍历 `configs` 字典中的 `layers` 列表，每个列表项代表一层。

3. 添加层：

   - 如果层类型为 `'dense'`，则添加一个全连接层（`Dense`）。

   - 如果层类型为 `'lstm'`，则添加一个 LSTM 层，并设置输入时间步和维度。

   - 如果层类型为 `'dropout'`，则添加一个 Dropout 层。

4. `load_model` 方法：

   - 这个方法用于从文件路径加载一个预训练的模型。
   - 它接收一个参数 `filepath`，表示模型文件的路径。
   - 使用 `load_model` 函数（需要从 `tensorflow.keras.models` 导入）来加载模型。

5. `build_model` 方法：
   - 这个方法用于根据配置构建模型。
   - 它接收一个参数 `configs`，这是一个包含模型配置的字典。
   - 方法内部使用了一个 `Timer` 类（需要从 `time` 模块导入）来测量模型构建的时间。
   

```py
import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from core.utils import Timer
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

class Model():
	"""LSTM 模型"""

	def __init__(self):
		self.model = Sequential()

	def load_model(self, filepath):
		print('[Model] Loading model from file %s' % filepath)
		self.model = load_model(filepath)

	def build_model(self, configs):
		timer = Timer()
		timer.start()

		for layer in configs['model']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None

			if layer['type'] == 'dense':
				self.model.add(Dense(neurons, activation=activation))
			if layer['type'] == 'lstm':
				self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
			if layer['type'] == 'dropout':
				self.model.add(Dropout(dropout_rate))

		self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

		print('[Model] Model Compiled')
		timer.stop()
		
		return self.model
```

训练过程

```py 
	def train(self, x, y, epochs, batch_size, save_dir):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			EarlyStopping(monitor='val_loss', patience=2),
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
		]
		self.model.fit(
			x,
			y,
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks
		)
		self.model.save(save_fname)

		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()

	def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
		]
		self.model.fit_generator(
			data_gen,
			steps_per_epoch=steps_per_epoch,
			epochs=epochs,
			callbacks=callbacks,
			workers=1
		)
		
		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()
```

预测过程，点预测和全序列预测

```py
def predict_point_by_point(self, data):
		print('[Model] Predicting Point-by-Point...')
		predicted = self.model.predict(data)
		predicted = np.reshape(predicted, (predicted.size,))
		return predicted

def predict_sequences_multiple(self, data, window_size, prediction_len,debug=False):
    if debug == False:
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs
    else :
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data)/prediction_len)):
            print (data.shape)
            curr_frame = data[i*prediction_len]
            print (curr_frame)
            predicted = []
            for j in range(prediction_len):
                predict_result = self.model.predict(curr_frame[newaxis,:,:])
                print (predict_result)
                final_result = predict_result[0,0]
                predicted.append(final_result)
                curr_frame = curr_frame[1:]
                print (curr_frame)
                curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
                print (curr_frame)
            prediction_seqs.append(predicted)


def predict_sequence_full(self, data, window_size):
    print('[Model] Predicting Sequences Full...')
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
    return predicted

```

class Timer

```py
import datetime as dt

class Timer():

	def __init__(self):
		self.start_dt = None

	def start(self):
		self.start_dt = dt.datetime.now()

	def stop(self):
		end_dt = dt.datetime.now()
		print('Time taken: %s' % (end_dt - self.start_dt))
```

#### 徐华{#XH}

预处理步骤

```py
import pandas as pd
df = pd.read_csv('/private/数据/2019-2020sydatahb.csv')
df

data = df

import matplotlib.pyplot as plt
series = data.set_index(['date'], drop=True)
plt.figure(figsize=(10, 6))
series['沈阳过去1小时降水量(毫米)'].plot()
plt.show()
```

数据差分转换与归一化

```py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
import numpy as np
import math
 
# 数据的差分转换
def difference(data_set,interval=1):
    diff=list()
    for i in range(interval,len(data_set)):
        value=data_set[i]-data_set[i-interval]
        diff.append(value)
    return pd.Series(diff)
 
# 对预测的数据进行逆差分转换
def invert_difference(history,yhat,interval=1):
    return yhat+history[-interval]
 
# 将数据转换为监督学习集，移位后产生的NaN值补0
def timeseries_to_supervised(data,lag=1):
    df=pd.DataFrame(data)
    columns=[df.shift(i) for i in range(1,lag+1)]
    columns.append(df)
    df=pd.concat(columns,axis=1)
    df.fillna(0,inplace=True)
    return df
 
# 将数据缩放到[-1,1]之间
def scale(train,test):
    # 创建一个缩放器，将数据集中的数据缩放到[-1,1]的取值范围中
    scaler=MinMaxScaler(feature_range=(-1,1))
    # 使用数据来训练缩放器
    scaler=scaler.fit(train)
    # 使用缩放器来将训练集和测试集进行缩放
    train_scaled=scaler.transform(train)
    test_scaled=scaler.transform(test)
    return scaler,train_scaled,test_scaled
 
# 将预测值进行逆缩放，使用之前训练好的缩放器，x为一维数组，y为实数
def invert_scale(scaler,X,y):
    # 将X,y转换为一个list列表
    new_row=[x for x in X]+[y]
    # 将列表转换为数组
    array=np.array(new_row)
    # 将数组重构成一个形状为[1,2]的二维数组->[[10,12]]
    array=array.reshape(1,len(array))
    # 逆缩放输入的形状为[1,2]，输出形状也是如此
    invert=scaler.inverse_transform(array)
    # 只需要返回y值即可
    return invert[0,-1]
```

#### LSTM模型{#LSTM}

创建与训练

```py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_csv('/private/数据/2019-2020sydatahb.csv')  # 请替换为你的文件路径
data = df['沈阳过去1小时降水量(毫米)'].values.reshape(-1, 1)

# 数据缩放到[-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data)

# 将序列转换为监督学习问题
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # 输入序列 (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # 预测序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # 将所有的拼接在一起
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # 丢弃含有NaN值的行
    if dropnan:
        agg.dropna(inplace=True)
    return agg

n_hours = 64  # 使用3小时的数据预测下一个小时
n_features = 1  # 特征数量，这里只有降水量一个特征
reframed = series_to_supervised(data_scaled, n_hours, 1)

# 分割为训练集和测试集
values = reframed.values
n_train_hours = int(len(values) * 0.8)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# 分割为输入和输出
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# 重塑成3D形状 [样本, 时间步, 特征]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

```



```py
# 构建一个LSTM模型
def fit_lstm(train,batch_size,nb_epoch,neurons):
    # 将数据对中的x和y分开
    X,y=train[:,0:-1],train[:,-1]
    # 将2D数据拼接成3D数据，形状为[N*1*1]
    X=X.reshape(X.shape[0],1,X.shape[1])
 
    model=Sequential()
    model.add(LSTM(neurons,batch_input_shape=(batch_size,X.shape[1],X.shape[2]),stateful=True))
    model.add(Dense(1))
 
    model.compile(loss='mean_squared_error',optimizer='adam')
    for i in range(nb_epoch):
        # shuffle是不混淆数据顺序
        his=model.fit(X,y,batch_size=batch_size,verbose=1,shuffle=False)
        # 每训练完一次就重置一次网络状态，网络状态与网络权重不同
        model.reset_states()
    return model
 
# 开始单步预测
import tensorflow as tf  # 确保导入tensorflow

def forecast_lstm(model, batch_size, X):
    # 确保X是一个Tensor
    X = tf.convert_to_tensor(X, dtype=tf.float32)  # 添加这行代码来转换X
    X = tf.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

 
# 读取数据，将日期和时间列合并，其他列删除，合并后的列转换为时间格式，设为索引
data['date']=pd.to_datetime(data['date'])
series=data.set_index(['date'],drop=True)
 
# 将原数据转换为二维数组形式，例如：
# [[4.6838],[4.6882],[4.7048]]
raw_value=series.values
# 将数据进行差分转换，例如[[4.6838],[4.6882],[4.7048]]转换为[[4.6882-4.6838],[4.7048-4.6882]]
diff_value=difference(raw_value,1)
#
# 将序列形式的数据转换为监督学习集形式，例如[[10],[11],[12],[13]]
# 在此将其转换为监督学习集形式：[[0,10],[10,11],[11,12],[12,13]]，
# 即前一个数作为输入，后一个数作为对应的输出
supervised=timeseries_to_supervised(diff_value,1)
supervised_value=supervised.values
 
# 将数据集分割为训练集和测试集，设置后1000个数据为测试集
testNum=3000
train,test=supervised_value[:-testNum],supervised_value[-testNum:]
 
# 将训练集和测试集都缩放到[-1,1]之间
scaler,train_scaled,test_scaled=scale(train,test)
 
# 构建一个LSTM模型并训练，样本数为1，训练次数为5，LSTM层神经元个数为4
lstm_model=fit_lstm(train_scaled,1,1,4)
# 遍历测试集，对数据进行单步预测
predictions=list()
for i in range(len(test_scaled)):
    # 将测试集拆分为X和y
    X,y=test[i,0:-1],test[i,-1]
    # 将训练好的模型、测试数据传入预测函数中
    yhat=forecast_lstm(lstm_model,1,X)
    # 将预测值进行逆缩放
    yhat=invert_scale(scaler,X,yhat)
    # 对预测的y值进行逆差分
    yhat=invert_difference(raw_value,yhat,len(test_scaled)+1-i)
    # 存储正在预测的y值
    predictions.append(yhat)
 
# 计算方差
rmse=mean_squared_error(raw_value[-testNum:],predictions)
print("Test RMSE:",rmse)
plt.plot(raw_value[-testNum:])
plt.plot(predictions)
plt.legend(['true','pred'])
plt.show()
```

```py
# 模型训练
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# 绘制历史数据
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# 做出预测
yhat = model.predict(test_X)
test_X_reshaped = test_X.reshape((test_X.shape[0], n_hours*n_features))

# 逆缩放预测值
inv_yhat = np.concatenate((yhat, test_X_reshaped[:, -(n_features-1):]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# 逆缩放真实值
test_y_reshaped = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y_reshaped, test_X_reshaped[:, -(n_features-1):]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# 计算 RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
```

绘制结果

```py
import matplotlib.pyplot as plt

# 绘制实际降水量与预测降水量的对比图
plt.figure(figsize=(10, 6))
plt.plot(inv_y, label='Actual', color='blue')  # 实际降水量
plt.plot(inv_yhat, label='Predicted', color='red', linestyle='--')  # 预测降水量
plt.title('LSTM Model Precipitation Prediction vs Actual')
plt.xlabel('Time')
plt.ylabel('Precipitation Amount (mm)')
plt.legend()
plt.show()
```

```py
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np


#转成有监督数据
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    #数据序列(也将就是input) input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        #预测数据（input对应的输出值） forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    #拼接 put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # 删除值为NAN的行 drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


##数据预处理 load dataset
dataset = read_csv('data_set/pollution.csv', header=0, index_col=0)
values = dataset.values
#标签编码 integer encode direction
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
#保证为float ensure all data is float
values = values.astype('float32')
#归一化 normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
#转成有监督数据 frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
#删除不预测的列 drop columns we don't want to predict
reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
print(reframed.head())

#数据准备
#把数据分为训练数据和测试数据 split into train and test sets
values = reframed.values
#拿一年的时间长度训练
n_train_hours = 365 * 24
#划分训练数据和测试数据
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
#拆分输入输出 split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
#reshape输入为LSTM的输入格式 reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print ('train_x.shape, train_y.shape, test_x.shape, test_y.shape')
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

##模型定义 design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
#模型训练 fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=36, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
#输出 plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

#进行预测 make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
#预测数据逆缩放 invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
inv_yhat = np.array(inv_yhat)
#真实数据逆缩放 invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

#画出真实数据和预测数据
pyplot.plot(inv_yhat,label='prediction')
pyplot.plot(inv_y,label='true')
pyplot.legend()
pyplot.show()

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
```

#### CEEMDAN{#CEEMDAN}

预处理阶段

```py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

from tensorflow.keras import Sequential,layers,losses,utils,Input
from tensorflow.keras.layers import Dense,LSTM,Dropout,concatenate,Flatten, Conv1D, MaxPooling1D,Activation,RepeatVector,TimeDistributed
import tensorflow as tf
# from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Bidirectional,GRU,Lambda,Dot,Concatenate
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error,mean_absolute_percentage_error
from keras.layers import Bidirectional,LSTM,GRU,Lambda,Dot,Concatenate

import matplotlib.pyplot as plt
import matplotlib

from sklearn.decomposition import PCA
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras import optimizers
from keras.models import load_model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data=pd.read_csv('/private/数据/2019-2022sydatag2.csv')

col=[ 'date', 'Precipitation', 'Temperature', 'Humidity',
       'Barometric pressure', 'wind speed', 'visibility']
datacol=data[col]
datacol.isnull().sum()
# 直接删除缺失值
datashan=datacol.dropna()
features = ['Precipitation']
target = ['Precipitation']
X = data[features].values
y = data[target].values
test_split=round(len(X)*0.20)
df_for_training=X[:-7008]
df_for_testing=X[-7008:]

def createXY(dataset,n_past):
  dataX = []
  dataY = []
  for i in range(n_past, len(dataset)):
          dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
          dataY.append(dataset[i,0])
  return np.array(dataX),np.array(dataY)

trainX,trainY=createXY(df_for_training_scaled,48)
testX,testY=createXY(df_for_testing_scaled,48)
```

修改后的LSTM

```py
from keras.layers import LSTM, Dense, Dropout

from keras.layers import LSTM, Dense, Dropout

# 定义修改后的LSTM模型
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# 训练模型
history = model.fit(trainX, trainY, epochs=70, batch_size=16, validation_data=(testX, testY), verbose=2)
```

CEEMDAN分解

```py
# 导入相应的库
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import datetime
import warnings
import scipy
from scipy import stats
import statsmodels
from PyEMD import CEEMDAN, Visualisation
X1 = data['Precipitation']
```

分解结果

```py
#Decomposition results of RV for stock indices.

ceemdan = CEEMDAN()
ceemdan.ceemdan(X1.values.reshape(-1))
imfs_close, res_close = ceemdan.get_imfs_and_residue()

t = np.arange(0, len(X1), 1)
vis = Visualisation()
vis.plot_imfs(imfs=imfs_close, residue=res_close, t=t, include_residue=True)
# vis.plot_instant_freq(t, imfs=imfs)
vis.show()
decompose_data = pd.DataFrame(np.vstack((imfs_close, res_close)).T,columns = ['IMF%d'%(i+1) for i in range(len(imfs_close))] + ['Res'])
```
使用分解结果训练
```py
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import math
# import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.models import Sequential

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 定义时间序列划分函数
def TimeSeries(dataset, start_index, history_size, end_index, step,
               target_size, point_time, true):
    data = []  # 保存特征数据
    labels = []  # 保存特征数据对应的标签值

    start_index = start_index + history_size  # 第一次的取值范围[0:start_index]

    # 如果没有指定滑动窗口取到哪个结束，那就取到最后
    if end_index is None:
        # 数据集最后一块是用来作为标签值的，特征不能取到底
        end_index = len(dataset) - target_size

    # 滑动窗口的起始位置到终止位置每次移动一步
    for i in range(start_index, end_index):

        index = range(i - history_size, i, step)  # 第一次相当于range(0, start_index, 6)

        # 根据索引取出所有的特征数据的指定行
        data.append(dataset.iloc[index])
        # 用这些特征来预测某一个时间点的值还是未来某一时间段的值
        if point_time is True:  # 预测某一个时间点
            # 预测未来哪个时间点的数据，例如[0:20]的特征数据（20取不到），来预测第20个的标签值
            labels.append(true[i + target_size])

        else:  # 预测未来某一时间区间
            # 例如[0:20]的特征数据（20取不到），来预测[20,20+target_size]数据区间的标签值
            labels.append(true[i:i + target_size])

    # 返回划分好了的时间序列特征及其对应的标签值
    return np.array(data), np.array(labels)


# 按照7:2:1划分训练验证测试集
def get_tain_val_test(serie_data, window_size):
    train_num = int(len(serie_data) * 0.7)
    val_num = int(len(serie_data) * 0.9)  # 取2w-2.3w的数据用于验证
    history_size = window_size  # 每个滑窗取5-26-272天的数据量(表示短期中期长期预测)
    target_size = 0  # 预测未来下一个时间点的气温值
    step = 1  # 步长为1取所有的行

    # 求训练集的每个特征列的均值和标准差
    feat_mean = serie_data.mean(axis=0)
    feat_std = serie_data.std(axis=0)

    # 对整个数据集计算标准差
    feat = (serie_data - feat_mean) / feat_std

    # 构造训练集
    x_train, y_train = TimeSeries(dataset=serie_data, start_index=0, history_size=history_size, end_index=train_num,
                                  step=step, target_size=target_size, point_time=True, true=serie_data)

    # 构造验证集
    x_val, y_val = TimeSeries(dataset=serie_data, start_index=train_num, history_size=history_size, end_index=val_num,
                              step=step, target_size=target_size, point_time=True, true=serie_data)

    # 构造测试集
    x_test, y_test = TimeSeries(dataset=serie_data, start_index=val_num, history_size=history_size,
                                end_index=len(serie_data),
                                step=step, target_size=target_size, point_time=True, true=serie_data)
    return x_train, y_train, x_val, y_val, x_test, y_test

```

开始训练

```py
from tensorflow.keras.optimizers import Adam

y_pre_list = []
y_test_list = []
window_size = 12

for column in decompose_data.columns:
    serie_data = decompose_data[column]
    x_train, y_train, x_val, y_val, x_test, y_test = get_tain_val_test(serie_data, window_size)
    
    # 在每次循环开始时重新实例化模型
    model = Sequential([
        GRU(50, input_shape=(window_size, 1), return_sequences=False),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val), batch_size=16, verbose=1)
    
    y_pre = model.predict(x_test)
    y_pre_list.append(y_pre)
    y_test_list.append(y_test)

# 这里可以添加代码以评估模型性能，比如计算每一列的MSE或MAE
```

训练结果

```py
fig = plt.figure(figsize=(10,20))
for i,column in enumerate(decompose_data.columns):
    
    axes = fig.add_subplot(len(decompose_data.columns),1,i+1)
    axes.plot(y_test_list[i], 'b-',linewidth = '1', label='actual')
    axes.plot(y_pre_list[i], 'r-', linewidth = '1', label='predict')

    plt.legend()
```

预测求和

```py
#此时的预测是对全部分解结果的预测求和
y_pre_total = np.sum(np.array(y_pre_list),axis = 0).reshape(-1)
x_train_all,y_train_all,x_val_all, y_val_all,x_test_all, y_test_all = get_tain_val_test(X1,window_size)
```

绘制加和结果

```py
fig = plt.figure(figsize=(10,5))
axes = fig.add_subplot(111)
axes.plot(y_test_all, 'b-',linewidth = 1, label='actual')
axes.plot(y_pre_total, 'r-', linewidth = 1, label='predict')

plt.legend()
#plt.grid()
plt.show()
```

一些结果误差

```py
def mean_absolute_error(y_test,y_pre):
    mae = np.sum(np.absolute(y_pre-y_test))/len(y_test)
    return mae
def mean_squared_error(y_test,y_pre):
    mse = np.sum((y_pre-y_test)**2)/len(y_test)
    return mse
def h_mean_absolute_error(y_test,y_pre):
    hmae = mean_absolute_error(y_test,y_pre) / np.mean(y_pre)
    return hmae
def h_mean_squared_error(y_test,y_pre):
    hmse = mean_squared_error(y_test,y_pre) / np.mean(y_pre) ** 2
    return hmse
from sklearn.metrics import r2_score
r2_train = r2_score(y_test_all, y_pre_total)

print("MAE:", mean_absolute_error(y_test_all, y_pre_total))
print("MSE:", mean_squared_error(y_test_all, y_pre_total))
print("HMAE:", h_mean_absolute_error(y_test_all, y_pre_total))
print("HMSE:", h_mean_squared_error(y_test_all, y_pre_total))
print(f'R^2: {r2_train}')
# 将预测结果中小于0的值设置为0
y_pre_total[y_pre_total < 0] = 0
fig = plt.figure(figsize=(10,5))
axes = fig.add_subplot(111)
axes.plot(y_test_all, 'b-',linewidth = 1, label='actual')
axes.plot(y_pre_total, 'r-', linewidth = 1, label='predict')

plt.legend()
#plt.grid()
plt.show()
```

相关性分析代码

```py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据集
data = pd.read_csv('/private/数据/2019-2020sydatahbg_E.csv')  # 替换为你的数据集路径

# 计算特征之间的相关性
correlation_matrix = data.corr()

# 可视化相关性矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Weather Features')
plt.show()
data = pd.read_csv('/private/数据/2019-2020sydatahbg.csv')  # 替换为你的数据集路径

col=['沈阳过去1小时降水量(毫米)', '温度/气温(摄氏度(℃))', '相对湿度(百分率)', '气压(百帕)',
       '10分钟平均风速(米/秒)', '10分钟平均水平能见度(m)']
datacol=data[col]
# Since the data is already loaded and 'day' is set as index in the previous step, 
# we directly calculate the correlation matrix here.

# Calculate the correlation matrix using Pearson's method
correlation_matrix_updated = data.corr()

# Display the correlation matrix
correlation_matrix_updated
correlation_matrix = data.corr()
correlation_matrix

import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy
warnings.filterwarnings("ignore")
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.sans-serif'] = ['SimHei']

# 设置热图的尺寸
plt.figure(figsize=(10, 6))

# 绘制热图
sns.heatmap(correlation_matrix_updated, annot=True, cmap='coolwarm', fmt=".2f")

# 添加标题
plt.title('Correlation Heatmap')

# 显示图形
plt.show()
```

