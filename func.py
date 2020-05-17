import numpy as np

def ReLU(x):
    return np.maximum(0, x)

class Conv3x3:
    # A convolution layer using 3x3 filters
    def __init__(self, fN, fC, fH, fW):
        # filter is a 4d array with dimensions (fN, 1, 3, 3)
        # We divide by 9 to reduce the variance of our initial values
        self.fN = fN
        self.fC = fC
        self.fH = fH
        self.fW = fW
        self.weights = np.random.randn(fN, fC, fH, fW) / (fH*fW)
        self.bias = np.zeros((fN, fC))
        #self.num_filters = np.array([ [0.1, 0.1, 0.1],
        #                              [0.0, 0.0, 0.0],
        #                              [-0.1, -0.1, -0.1]])

    #def iterate_region(self, image):

    def forward(self, input, stride=1, padding=1):
        # 28x28
        # self.last_input = input
        if np.ndim(input) != 4:
            print("input array dimension is not 4, program exit!!!")
            exit()

        # input_im: matrix of image
        self.iN, self.iC, self.iH, self.iW = input.shape
        self.oN = self.iN
        self.oC = self.fN
        self.oH = ((self.iH-self.fH+2*padding)//stride + 1)
        self.oW = ((self.iW-self.fW+2*padding)//stride + 1)
        output = np.zeros((self.oN, self.oC, self.oH, self.oW))

        # padding
        tmp = np.zeros((self.iN, self.iC, self.iH + 2, self.iW + 2))
        for n in range(self.iN):
            for ic in range(self.iC):
                tmp[n, ic, 1:-1, 1:-1] = input[n, ic, :, :]
        self.last_input = tmp

        self.imgOut = np.zeros((self.oN, self.oC, self.oH, self.oW))
        for n in range(self.iN):                            # 遍历iN个输入的图片
            for fn in range(self.fN):                       # 遍历fN个滤波器
                for h in range(self.oH):                    # 遍历卷积操作的列
                    for w in range(self.oW):                # 遍历卷积操作的行
                        for ic in range(self.iC):           # 遍历对应行列卷积的图片channel
                            tmp = self.last_input[n, ic, h:h + self.fH, w:w + self.fW] * self.weights[fn, ic, :, :]
                            self.imgOut[n, fn, h, w] += ReLU(np.sum(tmp) + self.bias[fn, ic])

        #return self.imgOut

    '''
    def backward(self, gradOut, learn_rate):
        # gradOut: the loss gradient for this layer's outputs
        # learn_rate: a float
        grad = {}
        grad['W'] = np.zeros_like(self.weights)
        grad['b'] = np.zeros_like(self.bias)
        grad['a'] = np.zeros_like(self.last_input)
        for n in range(self.iN):                            # 遍历iN个输入的图片
            for fn in range(self.fN):                       # 遍历fN个滤波器
                for h in range(self.oH):                    # 遍历卷积操作的列
                    for w in range(self.oW):                # 遍历卷积操作的行
                        for ic in range(self.iC):           # 遍历对应行列卷积的图片channel
        for h in range(self.iH):
            for w in range(self.iW):
                grad['W'] += gradOut * self.last_input[]
    '''






