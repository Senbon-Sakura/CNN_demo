import mnist
import numpy as np
import matplotlib.pyplot as plt
import func

# get dataSet and trimming dataset number
trainData, trainLabel, testData, testLabel = mnist.load_mnist(normalize=False, flatten=False, one_hot_label=False)
# mnist dataset:
    # trainData.shape = (60000, 1, 28, 28)
    # trainLabel.shape = (60000,)
    # testData.shape = (10000, 1, 28, 28)
    # testLabel.shape = (10000,)
# decrease traing/test figure number for saving time
trainData = trainData[:10]
trainLabel = trainLabel[:10]
testData = testData[:10]
testLabel = testLabel[:10]


conv = func.Conv3x3(3, 1, 3, 3)     # fN, fC, fH, fW
conv.forward(trainData)

# 显示图片
plt.subplot(2,2,1)
plt.imshow(trainData[1].reshape(28,28), cmap='gray')
plt.axis('off')
plt.subplot(2,2,2)
plt.imshow(conv.imgOut[1,0,:,:].reshape(28,28), cmap='gray')
plt.axis('off')
plt.subplot(2,2,3)
plt.imshow(conv.imgOut[1,1,:,:].reshape(28,28), cmap='gray')
plt.axis('off')
plt.subplot(2,2,4)
plt.imshow(conv.imgOut[1,2,:,:].reshape(28,28), cmap='gray')
plt.axis('off')
plt.show()

