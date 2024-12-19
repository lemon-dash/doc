## 其他行业

[源代码(李海军)](../../lhj.zip)

[源代码(靳智怡)](../../jzy.zip)

[源代码(resnet)](../../resnet.zip)

[源代码(senet)](../../senet.zip)

[源代码(sknet)](../../sknet.zip)

数据处理

```py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# set [ value & label ] align
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

# read data
df = pd.read_csv('./data.csv')

# show data
print(df.to_string())
((df.isnull().sum())/df.shape[0]).sort_values(ascending=False).map(lambda x:"{:.2%}".format(x))
# For feature '长' and '宽', it's same in every samples, so drop it.
df = df.drop("长", axis=1)
df = df.drop("宽", axis=1)

# Show changed data
print(df.to_string())
df["类型"] = df["类型"].replace({"木": 0, "铸铁": 1})
df["漏斗类型"] = df["漏斗类型"].replace({"圆形": 0, "椭圆形": 1, "椭圆": 1})
df.fillna(0, inplace=True)
print(df.to_string())
# Change data type:'  From object to the type below.
df['漏斗类型'] = pd.to_numeric(df['漏斗类型'], errors='coerce')
print(df.dtypes)
```

归一化

```py
# Create MinMaxScaler
# Use Max-Min-Normalization
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Show data
print(df_normalized.to_string())
x = df_normalized.drop(columns='漏斗类型')
y = df_normalized['漏斗类型']
# segment data to four part -> train & test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

```py
# Save data to ../set
x_train.to_csv('../set/x_train.csv', index=False)
x_test.to_csv('../set/x_test.csv', index=False)
y_train.to_csv('../set/y_train.csv', index=False)
y_test.to_csv('../set/y_test.csv', index=False)
# 计算相关系数矩阵
correlation_matrix = df_normalized.corr(method='pearson')
print(correlation_matrix.to_string())
import matplotlib

# 设置字体为支持中文的字体
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('皮尔逊相关系数热图')
plt.show()
```

#### MINST{#minst}

这段代码是用于训练一个简单的卷积神经网络（CNN）模型，目的是识别手写数字（MNIST数据集）

```py
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

model.save('digit_recognition_model.h5')
```

训练过程

```py
import numpy as np
import cv2
from PIL import ImageGrab
from tensorflow.keras.models import load_model

model = load_model('digit_recognition_model.h5')
def capture_screen(region):
    screen = ImageGrab.grab(bbox=region)
    # screen = Image.open("./image.png")
    return np.array(screen)


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return processed


def find_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def extract_digits(image, contours):
    digit_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 20 and h < 100: 
            roi = image[y : y + h, x : x + w]
            digit_images.append((x, roi))

    digit_images = sorted(digit_images, key=lambda item: item[0])
    return [digit[1] for digit in digit_images]


def recognize_digit(image):
    resized = cv2.resize(image, (28, 28))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    prediction = model.predict(reshaped)
    return np.argmax(prediction)


def recognize_digits(image):
    contours = find_contours(image)
    digit_images = extract_digits(image, contours)
    recognized_digits = [recognize_digit(digit) for digit in digit_images]
    return "".join(map(str, recognized_digits))

import pyautogui
def getDigital(region):
    screenshot = capture_screen(region)
    preprocessed_image = preprocess_image(screenshot)
    recognized_digits = recognize_digits(preprocessed_image)
    return recognize_digit


def draw_greater_than(start_x, start_y):
    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown()
    pyautogui.mouseDown()
    pyautogui.moveTo(start_x + 100, start_y - 100, duration=0.1)  # 右上
    pyautogui.moveTo(start_x + 100, start_y + 100, duration=0.1)  # 右下
    pyautogui.moveTo(start_x, start_y, duration=0.3)  # 左上
    pyautogui.mouseUp()


def draw_less_than(start_x, start_y):
    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown()
    pyautogui.mouseDown()
    pyautogui.moveTo(start_x - 100, start_y - 100, duration=0.1)  # 左上
    pyautogui.moveTo(start_x - 100, start_y + 100, duration=0.1)  # 左下
    pyautogui.moveTo(start_x, start_y, duration=0.3)  # 右上
    pyautogui.mouseUp()
    
while(True):
    region_1 = (247, 326, 292, 386)
    region_2 = (399, 323, 474, 395)
    num_1 = recognize_digit(region_1)
    num_2 = recognize_digit(region_2)
    print(num_1)
    print(num_2)
    if num_1 > num_2:
        draw_greater_than(333, 482)
    else:
        draw_less_than(333, 482)
```

