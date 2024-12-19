## 农业生鲜类

[源代码下载](../../agriculture.zip)

1. [李元浩](#LYH)

	1. [线性预测](#lf)
	1. [决策树](#dt)
	1. [随机森林](#rf)
	1. [adaboost](#ab)
	1. [GBRT](#gbrt)
	1. [XGBoost](#xb)
	1. [lightGBM](#lg)
	1. [SVR](#svr)
	1. [lasso回归](#las)
	1. [LSTM](#lstm)
	1. [CNN-GRU-AE](#cga)

2. 刘梦雅

 	1. [Pso_Cnn_BiLSTM_MultAE](#pcbma)
 	2. [woa_CNN_BiLSTM_MultAE](#woa)
 	3. [ssa_CNN_BiLSTM_MultAE](#ssa)
 	4. [BiLSTM_MultAE](#blma)
 	5. [CNN-MultAE](#cma)



**机器学习**

数据预处理

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
from tensorflow.keras.layers import Bidirectional,GRU,Lambda,Dot,Concatenate
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error,mean_absolute_percentage_error

import matplotlib.pyplot as plt
import matplotlib

from sklearn.decomposition import PCA

data=pd.read_csv('changping.csv')
col=[ 'year', 'month', 'day', 'hour', 'PM2.5', 'PM10', 'SO2', 'NO2',
       'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN',  'WSPM','wd']
datacol=data[col]
datacol.isnull().sum()
# 直接删除缺失值
datashan=datacol.dropna()
# 拼接时间为一列，绘制一年时间降水分布图
# datag['时间(年月日时)']=datag['年(年)'].map(str)+''+datag['月(月)'].map(str)+''+datag['日(日)'].map(str)+''+datag['时(时)'].map(str)
datashan['datatime']=datashan['year'].map(str)+'-'+datashan['month'].map(str)+'-'+datashan['day'].map(str)+'-'+datashan['hour'].map(str)

# datag=data.copy()
datashan['datatime']=pd.to_datetime(datashan['datatime'],format='%Y-%m-%d-%H',errors='coerce')
datashan.index=datashan['datatime']

chongcol=[ 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO',
       'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
datag=datashan[chongcol]
# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))  # minmaxscaler 方法，用来做归一化
sel_col = datag.columns
# print(sel_col)
for col in sel_col:
    datag[col] = scaler.fit_transform(datag[col].values.reshape(-1, 1))  # reshape成为一列
print("Load dataset LEN: ", datag.shape[0])
print(sel_col)
```

PCA降维

```py
X = []
Y = []
columns_name = list(datag.columns)
y_index = columns_name.index('PM2.5')
tmp = np.array(datag, dtype=np.float32)
print("The shape of tmp is:")
print(tmp.shape)
pca = PCA(n_components=0.97)# 保证降维后的数据保持90%的信息
tmp2 = pca.fit_transform(tmp[:,1:])
print("The shape of tmp2 is:")
print(tmp2.shape)  # PCA降维y_index要改，或者先提取出来
print("input dim is:")
print(tmp2.shape[1])
```

划分数据集

```py
seqX=120
seqY=1
day_delay=1
gap=1
X1=[]
for i in range(tmp2.shape[0] - (seqX+seqY+day_delay)*gap):
        # seqX, seqX+day_delay, seqX+day_delay+seqY
        # format: i+x*gap
        X.append(tmp2[i:(i + seqX*gap):gap, :])  # tmp2是PCA降维后的特征，如果是tmp那就是原始特征
        X1.append(tmp2[(i+(seqX+day_delay)*gap): (i+(seqX+day_delay+seqY)*gap): gap,:])
        Y.append(tmp[(i+(seqX+day_delay)*gap): (i+(seqX+day_delay+seqY)*gap): gap, y_index])
X = np.array(X)
Y = np.array(Y)
X1=np.array(X1)

shendu_trainx, shendu_trainy = X[:int(0.8 * 32559)], Y[:int(0.8 * 32559)]  
jiqi_trainx, jiqi_trainy = X1[:int(0.8 * 32559)], Y[:int(0.8 * 32559)]  
shendu_testx, shendu_testy = X[int(0.8 * 32559):], Y[int(0.8 * 32559):] 
jiqi_testx, jiqi_testy = X1[int(0.8 * 32559):], Y[int(0.8 * 32559):] 
#转化成机器学习可用数据
realjiqi_trainx=jiqi_trainx.reshape(26047,7)
realjiqi_testx=jiqi_testx.reshape(6512,7)

# 保存深度学习训练测试数据
# 将训练集存起来
nptrainx=np.reshape(shendu_trainx,(-1,7))
pdtrainx1=np.reshape(nptrainx,(-1,120,7))
# 保存深度学习训练测试数据
# 将训练集存起来
nptrainx=np.reshape(shendu_trainx,(-1,7))
pdtrainx1=np.reshape(nptrainx,(-1,120,7))
pdshendux_test=pd.DataFrame(nptestx,columns=['t1','t2','t3','t4','t5','t6','t7'])
pdshendux_test.to_csv('shendux_test.csv',index=False)
pdshenduy_train=pd.DataFrame(shendu_trainy,columns=['y1'])
pdshenduy_train.to_csv('shenduy_train.csv',index=False)
pdshenduy_test=pd.DataFrame(shendu_testy,columns=['y1'])
pdshenduy_test.to_csv('shenduy_test.csv',index=False)
# 保存机器学习训练测试集
pdjiqitrainx=pd.DataFrame(realjiqi_trainx,columns=['t1','t2','t3','t4','t5','t6','t7'])
pdjiqitrainx.to_csv('jiqix_train.csv',index=False)
pdjiqitestx=pd.DataFrame(realjiqi_testx,columns=['t1','t2','t3','t4','t5','t6','t7'])
pdjiqitestx.to_csv('jiqix_test.csv',index=False)
pdjiqiy_train=pd.DataFrame(jiqi_trainy,columns=['y1'])
pdjiqiy_train.to_csv('jiqiy_train.csv',index=False)
pdjiqiy_test=pd.DataFrame(jiqi_testy,columns=['y1'])
pdjiqiy_test.to_csv('jiqiy_test.csv',index=False)
```

读取深度学习数据

```py
# 读取训练集
dushendux_train=pd.read_csv('shendux_train.csv')
dushendux_train1=np.array(dushendux_train)
dushendux_train2=np.reshape(dushendux_train1,(-1,120,7))

dushenduy_train=pd.read_csv('shenduy_train.csv')
dushenduy_train2=np.array(dushenduy_train)
# 读取测试集
dushendux_test=pd.read_csv('shendux_test.csv')
dushendux_test1=np.array(dushendux_test)
dushendux_test2=np.reshape(dushendux_test1,(-1,120,7))

dushenduy_test=pd.read_csv('shenduy_test.csv')
dushenduy_test2=np.array(dushenduy_test)
```

读取数据机器学习

```py
# 读取训练集
dujiqix_train=pd.read_csv('jiqix_train.csv')
dujiqiy_train=pd.read_csv('jiqiy_train.csv')
# 读取测试集
dujiqix_test=pd.read_csv('jiqix_test.csv')
dujiqiy_test=pd.read_csv('jiqiy_test.csv')
```

#### 线性预测{#lf}

```py
from sklearn.linear_model import LinearRegression,RidgeCV,SGDRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score   #均方误差
# # # # 1.线性回归----此处使用基模型，参数均使用默认的，用于后面调参对比
# Lr=LinearRegression(normalize=False)
# Lr.fit(dujiqix_train,dujiqiy_train)
# xy_predict=Lr.predict(dujiqix_test)
# xy_modify = xy_predict*(xy_predict>=0)
# r2=r2_score(dujiqiy_test,xy_modify)
# # Mae=mean_squared_error(y_test,xy_modify)
# r2
 # 数据逆归一化
maxmin = [datal['PM2.5'].max(), datal['PM2.5'].min()]  # 原来的最大最小值，反放缩
print(maxmin)
# xpreds = np.array(xy_modify)  # 转换为numpy形式
# xlabels = np.array(dujiqiy_test)
# xpreds= xpreds.reshape((-1,1))
# xfuture_len= xpreds.shape[1]  # 获取要预测的天数长度，8预测5的话那就是5
# for k in range(xfuture_len):
#     xlabels[:, k] = xlabels[:, k] * (maxmin[0] - maxmin[1]) + maxmin[1]  # 反归一化
# for k in range(xfuture_len):
#     xpreds[:, k] = xpreds[:, k] * (maxmin[0] - maxmin[1]) + maxmin[1]  # 反归一化
```

#### 决策树{#dt}

```py
# tree_model=DecisionTreeRegressor()
# tree_model.fit(dux_train,duy_train)
# treey_pre=tree_model.predict(dux_test)
# ty_modify = treey_pre*(treey_pre>=0)
# jr=r2_score(duy_test,ty_modify)
# # treemae2=mean_squared_error(y_test,ty_modify2)
# # print(jr2,treemae2)
# jr
#  # 数据逆归一化
# maxmin = [datal['PM2.5'].max(), datal['PM2.5'].min()]  # 原来的最大最小值，反放缩
# print(maxmin)
# preds = np.array(ty_modify)  # 转换为numpy形式
# labels1 = np.array(duy_test)
# preds= preds.reshape((-1,1))
# future_len= preds.shape[1]  # 获取要预测的天数长度，8预测5的话那就是5
# for k in range(future_len):
#     labels1[:, k] = labels1[:, k] * (maxmin[0] - maxmin[1]) + maxmin[1]  # 反归一化
# for k in range(future_len):
#     preds[:, k] = preds[:, k] * (maxmin[0] - maxmin[1]) + maxmin[1]  # 反归一化
```

#### 随机森林{#rf}

```py
from sklearn.ensemble import RandomForestRegressor
rd=RandomForestRegressor(max_depth=11,max_features=6,min_samples_split=35,n_estimators=96,min_samples_leaf=9)     #基模型
rd.fit(dujiqix_train,dujiqiy_train)
suiy_pres=rd.predict(dujiqix_test)
yrand_modify = suiy_pres*(suiy_pres>=0)
suijir=r2_score(dujiqiy_test,yrand_modify)
label = np.array(dujiqiy_test)
for k in range(1):
#     print(k)
    label[:, k] = label[:, k] * (maxmin[0] - maxmin[1]) + maxmin[1]  # 反归一化
np.array(dujiqiy_test)
spreds = np.array(yrand_modify)  # 转换为numpy形式
# labels1 = np.array(dujiqiy_test)
spreds= spreds.reshape((-1,1))
sfuture_len= spreds.shape[1]  # 获取要预测的天数长度，8预测5的话那就是5
# for k in range(sfuture_len):
#     labels1[:, k] = labels1[:, k] * (maxmin[0] - maxmin[1]) + maxmin[1]  # 反归一化
for k in range(sfuture_len):
    spreds[:, k] = spreds[:, k] * (maxmin[0] - maxmin[1]) + maxmin[1]  # 反归一化
```

#### Adaboost{#ab}

```py
from sklearn.ensemble import AdaBoostRegressor
# 也可以对选择的弱回归器进行参数选择，方式为base_estimator__若回归器参数名
Ada_model=AdaBoostRegressor(DecisionTreeRegressor(max_depth=11, min_samples_split=35, min_samples_leaf=35),
                            n_estimators=50,learning_rate=0.1,loss='linear')
Ada_model.fit(dujiqix_train,dujiqiy_train)
aday_pres=Ada_model.predict(dujiqix_test)
aday_modify = aday_pres*(aday_pres>=0)
adar=r2_score(dujiqiy_test,aday_modify)
apreds = np.array(aday_modify)  # 转换为numpy形式
# labels1 = np.array(dujiqiy_test)
apreds= apreds.reshape((-1,1))
afuture_len= apreds.shape[1]  # 获取要预测的天数长度，8预测5的话那就是5
# for k in range(afuture_len):
#     labels1[:, k] = labels1[:, k] * (maxmin[0] - maxmin[1]) + maxmin[1]  # 反归一化
for k in range(afuture_len):
    apreds[:, k] = apreds[:, k] * (maxmin[0] - maxmin[1]) + maxmin[1]  # 反归一化
```

#### GBRT{#gbrt}

```py
from sklearn.ensemble import GradientBoostingRegressor
Gbrt_model= GradientBoostingRegressor()#这里使用50个决策树
Gbrt_model.fit(dujiqix_train,dujiqiy_train)
gbrty_pres=Gbrt_model.predict(dujiqix_test)
gbrty_modify = gbrty_pres*(gbrty_pres>=0)
gbrtr=r2_score(dujiqiy_test,gbrty_modify)
gpreds = np.array(gbrty_modify)  # 转换为numpy形式
# labels2 = np.array(dujiqiy_test)
gpreds= gpreds.reshape((-1,1))
gfuture_len= gpreds.shape[1]  # 获取要预测的天数长度，8预测5的话那就是5
# for k in range(gfuture_len):
#     labels2[:, k] = labels2[:, k] * (maxmin[0] - maxmin[1]) + maxmin[1]  # 反归一化
for k in range(gfuture_len):
    gpreds[:, k] = gpreds[:, k] * (maxmin[0] - maxmin[1]) + maxmin[1]  # 反归一化
```

#### XGBoost{#xb}

```py
from xgboost import XGBRegressor

other_params = {'learning_rate': 0.1, 'n_estimators': 75, 'max_depth': 4, 'min_child_weight': 5, 
                     'colsample_bytree': 0.7, 'gamma': 0.01,}
xgb_model1=XGBRegressor(**other_params)
xgb_model1.fit(dujiqix_train,dujiqiy_train)
xgby1_pred=xgb_model1.predict(dujiqix_test)
xgby1_modify = xgby1_pred*(xgby1_pred>=0)
xgbr1=r2_score(dujiqiy_test,xgby1_modify)

other_params1 = {'learning_rate': 0.1, 'n_estimators': 70, 'max_depth': 4, 'min_child_weight': 5, 
                     'colsample_bytree': 0.7, 'gamma': 0.01,
                 }
xgb_model=XGBRegressor(**other_params1)
xgb_model.fit(dujiqix_train,dujiqiy_train)
xgby_pred=xgb_model.predict(dujiqix_test)
xgby_modify = xgby_pred*(xgby_pred>=0)
xgbr=r2_score(dujiqiy_test,xgby_modify)
xgbpreds = np.array(xgby1_modify)  # 转换为numpy形式
# labels3 = np.array(dujiqiy_test)
xgbpreds= xgbpreds.reshape((-1,1))
xfuture_len= xgbpreds.shape[1]  # 获取要预测的天数长度，8预测5的话那就是5
# for k in range(xfuture_len):
#     labels3[:, k] = labels3[:, k] * (maxmin[0] - maxmin[1]) + maxmin[1]  # 反归一化
for j in range(xfuture_len):
    xgbpreds[:, j] = xgbpreds[:, j] * (maxmin[0] - maxmin[1]) + maxmin[1]  # 反归一化
```

#### lightGBM{#lg}

```py
from lightgbm import LGBMRegressor
import lightgbm as lgb
tmodel_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=36,
                              learning_rate=0.01, n_estimators=430, max_depth=8, 
                              metric='rmse', min_child_samples=4,
                            )#reg_alpha=0.5,reg_lambda=0.5,feature_fraction=0.6,bagging_fraction=0.4,
tmodel_lgb.fit(dujiqix_train,dujiqiy_train)
lgby=tmodel_lgb.predict(dujiqix_test)
lgby_modify=lgby*(lgby>=0)
lgbr=r2_score(dujiqiy_test,lgby_modify)

tmodel_lgb1 = lgb.LGBMRegressor(objective='regression',num_leaves=30,
                              learning_rate=0.01, n_estimators=500, max_depth=8, 
                              metric='rmse', min_child_samples=5,
                            )#reg_alpha=0.5,reg_lambda=0.5,feature_fraction=0.6,bagging_fraction=0.4,
tmodel_lgb1.fit(dujiqix_train,dujiqiy_train)
lgby1=tmodel_lgb1.predict(dujiqix_test)
lgby_modify1=lgby1*(lgby1>=0)
lgbr1=r2_score(dujiqiy_test,lgby_modify1)

lpreds1 = np.array(lgby_modify1)  # 转换为numpy形式
# llabel = np.array(dujiqiy_test)
lpreds1= lpreds1.reshape((-1,1))
lfuture_len= lpreds1.shape[1]  # 获取要预测的天数长度，8预测5的话那就是5
# for k in range(lfuture_len):
#     llabel[:, k] = llabel[:, k] * (maxmin[0] - maxmin[1]) + maxmin[1]  # 反归一化
for k in range(lfuture_len):
    lpreds1[:, k] = lpreds1[:, k] * (maxmin[0] - maxmin[1]) + maxmin[1]  # 反归一化
```

#### SVR{#svr}

```py
from sklearn.svm import SVR
svr_rbf = SVR(kernel='linear', C=10, gamma=1)
svr_rbf.fit(dujiqix_train,dujiqiy_train)
svry_pred= svr_rbf.predict(dujiqix_test)
svry_modify= svry_pred*(svry_pred>=0)

svrr=r2_score(dujiqiy_test,svry_modify)
svr_rbf2 = SVR(kernel='rbf', C=10, gamma=1)
svr_rbf2.fit(dujiqix_train,dujiqiy_train)
svry_pred2= svr_rbf2.predict(dujiqix_test)
svry_modify2 = svry_pred2*(svry_pred2>=0)
svrr2=r2_score(dujiqiy_test,svry_modify2)
svrpreds = np.array(svry_modify2)  # 转换为numpy形式
# labels1 = np.array(y_test)
svrpreds= svrpreds.reshape((-1,1))
svrfuture_len= svrpreds.shape[1]  # 获取要预测的天数长度，8预测5的话那就是5
# for k in range(sfuture_len):
#     labels1[:, k] = labels1[:, k] * (maxmin[0] - maxmin[1]) + maxmin[1]  # 反归一化
for k in range(svrfuture_len):
    svrpreds[:, k] = svrpreds[:, k] * (maxmin[0] - maxmin[1]) + maxmin[1]  # 反归一化
```

#### lasso回归{#las}

```py
from sklearn.linear_model import LassoCV 
La=LassoCV()
La.fit(dujiqix_train,dujiqiy_train)
lay_predict=La.predict(dujiqix_test)
lay_modify = lay_predict*(lay_predict>=0)
rla=r2_score(dujiqiy_test,lay_modify)
lapreds = np.array(lay_modify)  # 转换为numpy形式
# labels1 = np.array(y_test)
lapreds= lapreds.reshape((-1,1))
lafuture_len= lapreds.shape[1]  # 获取要预测的天数长度，8预测5的话那就是5
# for k in range(lafuture_len):
#     labels1[:, k] = labels1[:, k] * (maxmin[0] - maxmin[1]) + maxmin[1]  # 反归一化
for k in range(lafuture_len):
    lapreds[:, k] = lapreds[:, k] * (maxmin[0] - maxmin[1]) + maxmin[1]  # 反归一化
    
```

#### LSTM{#lstm}

```py
# 读取训练集
dushendux_train=pd.read_csv('shendux_train.csv')
dushendux_train1=np.array(dushendux_train)
dushendux_train2=np.reshape(dushendux_train1,(-1,120,7))

dushenduy_train=pd.read_csv('shenduy_train.csv')
dushenduy_train2=np.array(dushenduy_train)
# 读取测试集
dushendux_test=pd.read_csv('shendux_test.csv')
dushendux_test1=np.array(dushendux_test)
dushendux_test2=np.reshape(dushendux_test1,(-1,120,7))

dushenduy_test=pd.read_csv('shenduy_test.csv')
dushenduy_test2=np.array(dushenduy_test)

# # # 构造批数据
def create_batch_dataset(x,y,train=True,buffer_size=1000,batch_size=64):#buffer_size=1000表示可以打乱窗口里面的数据
    batch_data=tf.data.Dataset.from_tensor_slices((tf.constant(x),tf.constant(y)))#数据封装，tensor类型
    if train:
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else:
        return batch_data.batch(batch_size)
train_data_single=create_batch_dataset(dushendux_train2, dushenduy_train2,train=True)
val_data_single=create_batch_dataset(dushendux_test2,dushenduy_test2,train=False)

# 添加注意力机制
def attention_bilstm(inputs):
    """
        Many-to-one attention mechanism for Keras.
        @param inputs: 3D tensor with shape (batch_size, time_steps, input_dim).
        @return: 2D tensor with shape (batch_size, 128)
        @author: felixhao28, philipperemy.
        """
    hidden_states = inputs
    hidden_size = int(hidden_states.shape[2])
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    score = Dot(axes=[1, 2], name='attention_score')([h_t, score_first_part])
    attention_weights = Activation('softmax', name='attention_weight')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
    context_vector = Dot(axes=[1, 1], name='context_vector')([hidden_states, attention_weights])
    pre_activation = Concatenate(name='attention_output')([context_vector, h_t])
    attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return attention_vector

# 构造BI_GRU
# attbilstm_model = Sequential()
time_steps=120
input_dim=7
model_input=Input(shape=(time_steps,input_dim))
bi_gru1=Bidirectional(LSTM(128,return_sequences=True))(model_input)
bi_gru2=Dropout(0.2)(bi_gru1)
bi_gru3=Bidirectional(LSTM(64,return_sequences=True))(bi_gru2)
# attbilstm_model.add(Bidirectional(GRU(64), input_shape=trainx.shape[-2:],return_sequences=True))
attbigru=attention_bilstm(bi_gru3)
# a=Dropout(0.2)(attbigru)
zhong=Dense(1)(attbigru)
attbilstm_model=Model(model_input,zhong)
attbilstm_model.compile(optimizer='adam', loss='mae')# metrics=['accuracy']
attbilstm_history = attbilstm_model.fit(train_data_single, validation_data=val_data_single,epochs=800, verbose=1)

# 预测
attbilstm_pres=attbilstm_model.predict(dushendux_test2,verbose=1)
attbilstm_modify = attbilstm_pres*(attbilstm_pres>=0)
lstmr1=r2_score(dushenduy_test2,attbilstm_modify)
```

#### CNN-GRU-AE{#cga}

```py
# 设置时间为数据的索引
datacol.index = datacol['day']
blueBerry = ['hardness', 'L', 'a', 'b', 'detE', 'solubleSolid',
       'titrableAcid','weightLoss','rotRate','anthocyanin','totalPhenol','flavonoid','shelfLife']
datag = datacol[blueBerry]
datal = datacol[blueBerry]

# 特征选择后的数据
feature_selection = ['totalPhenol', 'titrableAcid', 'flavonoid', 'detE', 'rotRate', 'weightLoss','hardness','shelfLife']
datal = datal[feature_selection]

def prepare_data(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i + seq_length].values  # 获取连续的seq_length行，并转换为NumPy数组
        X.append(seq)
        y.append(data.iloc[i + seq_length].values)  # 获取下一行，并转换为NumPy数组作为标签
    return np.array(X), np.array(y)
# 转换为 PyTorch 张量
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float()

# 使用 DataLoader 封装数据
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

模型结构

- `Attention` 类是一个 PyTorch 的 `nn.Module` 子类，用于实现注意力机制。在初始化方法中，它创建了一个线性层，用于将隐藏状态和编码器输出的合并结果映射到一个中间维度。这里的 `input_size * 2` 应该是 `hidden_size * 2`，因为 `input_size` 在这里没有定义，而 `seq_length` 也没有在类中定义，它应该是序列的长度。

- `NN_GRU_Model` 类也是一个 `nn.Module` 子类，它定义了一个包含CNN层、GRU层和全连接层的模型，并添加了一个 `Attention` 层。在初始化方法中，定义了一个一维卷积层 `cnn_layer`，计算了卷积层的输出大小，定义了一个GRU层 `gru_layer`，一个全连接层 `fc`，以及一个注意力层 `attention`。

- 在 `forward` 方法中，首先将输入 `x` 的最后两个维度交换位置以适应卷积层的要求。然后，通过卷积层 `cnn_layer` 处理输入。接着，重新排列维度以适应GRU层的输入要求。通过注意力层 `attention` 计算注意力权重，并将这些权重应用到CNN的输出上以获得上下文向量 `context`。然后，将上下文向量通过GRU层 `gru_layer` 处理，并最终通过全连接层 `fc` 得到输出。

```py title="CNN-GRU-AE"
# 添加注意力机制
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(input_size * 2, seq_length)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        # h = hidden.repeat(1, seq_len, 1)
        h = hidden
        energy = torch.tanh(self.attn(torch.cat((h, encoder_outputs), dim=2)))
        attention_weights = F.softmax(energy, dim=1)
        context_vector = torch.bmm(attention_weights.transpose(1, 2), encoder_outputs)
        return context_vector

# 定义模型
class CNN_GRU_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size):
        super(CNN_GRU_Model, self).__init__()
        # 增加卷积层的输出通道数
        self.cnn_layer = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=kernel_size)
        self.conv_output_size = input_size - kernel_size + 1  # 计算卷积层输出大小
        self.gru_layer = nn.GRU(self.conv_output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

        # 添加注意力机制的层
        self.attention = Attention(hidden_size)

    def forward(self, x):
        # 转置输入以适应卷积层的要求
        x = x.transpose(1, 2)  # 将最后两个维度交换位置，从(2, 3, 14)变为(2, 14, 3)

        # CNN层
        x = self.cnn_layer(x)  # 卷积层输入形状为(batch_size, 特征数, 时间步长)
        x = x.permute(0, 2, 1)  # 重新排列维度，将序列长度放在第 1 维

        # 注意力机制
        attn_weights = self.attention(x, x)
        # 将注意力权重应用到CNN的输出上
        context = torch.bmm(attn_weights.transpose(1,2), x).squeeze(1)
        # GRU层
        _, h_n = self.gru_layer(context)

        # 全连接层
        out = self.fc(h_n[-1])

        return out
```

模型训练

```py
# 初始化模型
input_size = X.shape[2]  # 特征数
hidden_size = 64
num_layers = 1
kernel_size = 1
model = CNN_GRU_Model(input_size, hidden_size, num_layers, kernel_size)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 预测的值
with torch.no_grad():
    last_sequence = torch.from_numpy(data[-seq_length:].values).float().unsqueeze(0)
    prediction = model(last_sequence)
    print("预测结果:", prediction.numpy())

# 保存模型
torch.save(model.state_dict(), '1cnn_gru_ae_3_fs_model.pth')

```

#### Pso_Cnn_BiLSTM_MultAE{#pcbma}

```py
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, Dense, BatchNormalization, Flatten
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 输入形状
input_shape = (X_train_expanded.shape[1], 1)
inputs = Input(shape=input_shape)

# 卷积层
x = Conv1D(filters=13, kernel_size=3, strides=1, padding='valid')(inputs)
x = MaxPooling1D(pool_size=2)(x)
x = BatchNormalization()(x) 
# 双向LSTM层
x = Bidirectional(LSTM(37, return_sequences=True))(x)

# 多头注意力机制层
attention_output = MultiHeadAttention(num_heads=5, key_dim=74)(x, x)  # 注意力机制输入和输出应相同

# 将注意力输出展平
x = Flatten()(attention_output)

# Dropout层
x = Dropout(0.3)(x)

# 全连接层
outputs = Dense(1)(x)

# 模型定义和编译
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=0.00427), loss='mse', metrics=['mse'])

# 添加ModelCheckpoint回调函数
checkpoint = ModelCheckpoint('model/Pso_Cnn_BiLSTM_MultAE.model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# 训练模型
history = model.fit(X_train_expanded, y_train, validation_data=(X_val_expanded, y_val), epochs=300, batch_size=32, verbose=1, callbacks=[checkpoint])

# 载入保存的最佳模型
model.load_weights('model/Pso_Cnn_BiLSTM_MultAE.model.h5')

# 预测验证集
y_pred = model.predict(X_val_expanded).flatten()

# 计算MSE, RMSE, R2
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)

# 绘制训练和验证的损失曲线
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("MSE:", mse)
print("RMSE:", rmse)
print("R^2 Score:", r2)
```

预测

```py
# 保存模型
# model.save("model/Pso_Cnn_BiLSTM_MultAE.model.h5")
# print("模型已保存为Cnn_BiLSTM_MultAE.h5")
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score

# 加载保存的最佳模型
model = load_model('model/Pso_Cnn_BiLSTM_MultAE.model.h5')
# 将测试集数据增加一个维度以匹配BiLSTM的输入要求
X_test_expanded = np.expand_dims(X_test, axis=-1)

# 使用evaluate方法计算测试集上的损失和MSE
test_loss, test_mse = model.evaluate(X_test_expanded, y_test, verbose=1)

# 使用模型进行预测
y_test_pred = model.predict(X_test_expanded).flatten()

# 计算R^2分数
r2_test = r2_score(y_test, y_test_pred)

# 输出所有结果
print("Test Loss:", test_loss)
print("Test MSE:", test_mse)
print("R^2 Score on Test Set:", r2_test)
```

#### woa_CNN_BiLSTM_MultAE{#woa}

```py
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, Dense, BatchNormalization, Flatten
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 输入形状
input_shape = (X_train_expanded.shape[1], 1)
inputs = Input(shape=input_shape)

# 卷积层
x = Conv1D(filters=15, kernel_size=3, strides=1, padding='valid')(inputs)
x = MaxPooling1D(pool_size=2)(x)
x = BatchNormalization()(x) 
# 双向LSTM层
x = Bidirectional(LSTM(36, return_sequences=True))(x)

# 多头注意力机制层
attention_output = MultiHeadAttention(num_heads=4, key_dim=72)(x, x)  # 注意力机制输入和输出应相同

# 将注意力输出展平
x = Flatten()(attention_output)

# Dropout层
x = Dropout(0.3)(x)

# 全连接层
outputs = Dense(1)(x)

# 模型定义和编译
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=0.00303), loss='mse', metrics=['mse'])

# 训练模型
history = model.fit(X_train_expanded, y_train, validation_data=(X_val_expanded, y_val), epochs=300, batch_size=32, verbose=1)

# 预测验证集
y_pred = model.predict(X_val_expanded).flatten()

# 计算MSE, RMSE, R2
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)


# 绘制训练和验证的损失曲线
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("MSE:", mse)
print("RMSE:", rmse)
print("R^2 Score:", r2)

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score
# 保存模型
model.save("model/woa_Cnn_BiLSTM_MultAE.model.h5")
print("模型已保存为Cnn_BiLSTM_MultAE.h5")
# 加载保存的最佳模型
model = load_model('model/woa_Cnn_BiLSTM_MultAE.model.h5')
# 将测试集数据增加一个维度以匹配BiLSTM的输入要求
X_test_expanded = np.expand_dims(X_test, axis=-1)

# 使用evaluate方法计算测试集上的损失和MSE
test_loss, test_mse = model.evaluate(X_test_expanded, y_test, verbose=1)

# 使用模型进行预测
y_test_pred = model.predict(X_test_expanded).flatten()

# 计算R^2分数
r2_test = r2_score(y_test, y_test_pred)

# 输出所有结果
print("Test Loss:", test_loss)
print("Test MSE:", test_mse)
print("R^2 Score on Test Set:", r2_test)
```

#### ssa_CNN_BiLSTM_MultAE{#ssa}

```py
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, Dense, BatchNormalization, Flatten
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 输入形状
input_shape = (X_train_expanded.shape[1], 1)
inputs = Input(shape=input_shape)

# 卷积层
x = Conv1D(filters=15, kernel_size=3, strides=1, padding='valid')(inputs)
x = MaxPooling1D(pool_size=2)(x)
x = BatchNormalization()(x) 
# 双向LSTM层
x = Bidirectional(LSTM(37, return_sequences=True))(x)

# 多头注意力机制层
attention_output = MultiHeadAttention(num_heads=5, key_dim=74)(x, x)  # 注意力机制输入和输出应相同

# 将注意力输出展平
x = Flatten()(attention_output)

# Dropout层
x = Dropout(0.3)(x)

# 全连接层
outputs = Dense(1)(x)

# 模型定义和编译
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=0.00360), loss='mse', metrics=['mse'])

# 训练模型
history = model.fit(X_train_expanded, y_train, validation_data=(X_val_expanded, y_val), epochs=300, batch_size=32, verbose=1)

# 预测验证集
y_pred = model.predict(X_val_expanded).flatten()

# 计算MSE, RMSE, R2
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)


# 绘制训练和验证的损失曲线
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("MSE:", mse)
print("RMSE:", rmse)
print("R^2 Score:", r2)
```

预测

```py
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score
# 保存模型
model.save("model/ssa_Cnn_BiLSTM_MultAE.model.h5")
print("模型已保存为Cnn_BiLSTM_MultAE.h5")
# 加载保存的最佳模型
model = load_model('model/ssa_Cnn_BiLSTM_MultAE.model.h5')
# 将测试集数据增加一个维度以匹配BiLSTM的输入要求
X_test_expanded = np.expand_dims(X_test, axis=-1)

# 使用evaluate方法计算测试集上的损失和MSE
test_loss, test_mse = model.evaluate(X_test_expanded, y_test, verbose=1)

# 使用模型进行预测
y_test_pred = model.predict(X_test_expanded).flatten()

# 计算R^2分数
r2_test = r2_score(y_test, y_test_pred)

# 输出所有结果
print("Test Loss:", test_loss)
print("Test MSE:", test_mse)
print("R^2 Score on Test Set:", r2_test)
```

#### BiLSTM_MultAE{#blma}

```py
from keras.models import Model
from keras.layers import Input, Bidirectional, LSTM, Dropout, Dense, Flatten
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 输入形状
input_shape = (X_train_expanded.shape[1], 1)
inputs = Input(shape=input_shape)

# 双向LSTM层
x = Bidirectional(LSTM(32, return_sequences=True))(inputs)

# 多头注意力机制层
attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)

# 将注意力输出展平
x = Flatten()(attention_output)

# Dropout层
x = Dropout(0.3)(x)

# 全连接层
outputs = Dense(1)(x)

# 模型定义和编译
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])

# 训练模型
history = model.fit(X_train_expanded, y_train, validation_data=(X_val_expanded, y_val), epochs=300, batch_size=32, verbose=1)

# 预测验证集
y_pred = model.predict(X_val_expanded).flatten()

# 计算MSE, RMSE, R2
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)

# 绘制训练和验证的损失曲线
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("MSE:", mse)
print("RMSE:", rmse)
print("R^2 Score:", r2)
```

预测

```py
# 保存模型
model.save("model/BiLSTM_MultiHeadAttention.model.h5")
print("模型已保存为BiLSTM_MultiHeadAttention.h5")
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score

# 加载保存的最佳模型
model = load_model('model/BiLSTM_MultiHeadAttention.model.h5')
# 将测试集数据增加一个维度以匹配BiLSTM的输入要求
X_test_expanded = np.expand_dims(X_test, axis=-1)

# 使用evaluate方法计算测试集上的损失和MSE
test_loss, test_mse = model.evaluate(X_test_expanded, y_test, verbose=1)

# 使用模型进行预测
y_test_pred = model.predict(X_test_expanded).flatten()

# 计算R^2分数
r2_test = r2_score(y_test, y_test_pred)

# 输出所有结果
print("Test Loss:", test_loss)
print("Test MSE:", test_mse)
print("R^2 Score on Test Set:", r2_test)
```

#### CNN-MultAE{#cma}

```py
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.layers import MultiHeadAttention, Reshape
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 输入形状
input_shape = (X_train_expanded.shape[1], 1)
inputs = Input(shape=input_shape)

# 卷积层
x = Conv1D(filters=16, kernel_size=3, strides=1, padding='valid')(inputs)
x = MaxPooling1D(pool_size=2)(x)

# 调整卷积和池化后的输出以适应多头注意力机制，确保输出至少有两个维度
sequence_length = x.shape[1]
feature_dim = x.shape[2]
x = Reshape((sequence_length, feature_dim))(x)

# 多头注意力机制层
attention_output = MultiHeadAttention(num_heads=8, key_dim=feature_dim)(x, x)

# 展平处理注意力输出，准备全连接层
x = Flatten()(attention_output)

# Dropout层
x = Dropout(0.3)(x)

# 全连接层
outputs = Dense(1)(x)

# 模型定义和编译
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])

# 训练模型
history = model.fit(X_train_expanded, y_train, validation_data=(X_val_expanded, y_val), epochs=300, batch_size=32, verbose=1)

# 预测验证集
y_pred = model.predict(X_val_expanded).flatten()

# 计算MSE, RMSE, R2
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)

# 绘制训练和验证的损失曲线
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("MSE:", mse)
print("RMSE:", rmse)
print("R^2 Score:", r2)
```

预测

```py
# 保存模型
model.save("model/CNN_MultiHeadAttention.model.h5")
print("模型已保存为CNN_MultiHeadAttention.modelh5")

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score

# 加载保存的最佳模型
model = load_model('model/CNN_MultiHeadAttention.model.h5')
# 将测试集数据增加一个维度以匹配BiLSTM的输入要求
X_test_expanded = np.expand_dims(X_test, axis=-1)

# 使用evaluate方法计算测试集上的损失和MSE
test_loss, test_mse = model.evaluate(X_test_expanded, y_test, verbose=1)

# 使用模型进行预测
y_test_pred = model.predict(X_test_expanded).flatten()

# 计算R^2分数
r2_test = r2_score(y_test, y_test_pred)

# 输出所有结果
print("Test Loss:", test_loss)
print("Test MSE:", test_mse)
print("R^2 Score on Test Set:", r2_test)
```

