from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 定义输入形状，例如，64x64像素的图像，有3个通道（RGB）
input_shape = (64, 64, 3)

# 初始化模型
model = Sequential()

# 添加一个带有32个滤波器的卷积层，3x3的卷积核，以及ReLU激活函数
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

# 添加一个最大池化层，使用2x2的池化窗口以减少空间维度
model.add(MaxPooling2D(pool_size=(2, 2)))

# 再添加一个卷积层，滤波器数量加倍
model.add(Conv2D(64, (3, 3), activation='relu'))

# 再添加一个最大池化层
model.add(MaxPooling2D(pool_size=(2, 2)))

# 将卷积层的输出展平以便输入到密集层
model.add(Flatten())

# 添加一个全连接层，有128个单元和ReLU激活函数
model.add(Dense(128, activation='relu'))

# 添加输出层，有10个单元（对应10个类别），使用softmax激活函数
model.add(Dense(10, activation='softmax'))

# 编译模型，指定损失函数、优化器和要观察的指标
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 打印模型结构摘要
model.summary()
