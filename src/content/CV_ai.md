# 你對做影像辨識有興趣嗎
![是啊2](https://hackmd.io/_uploads/BJTkd8O5Jg.jpg)

## 什麼是影像辨識
影像辨識是一種利用人工智慧技術來分析和理解圖像內容的技術。影像辨識現在比較常見的應用像是在安全監控、醫療診斷、自動駕駛車輛、智能手機相機、工業自動化等。

影像辨識的核心技術包括卷積神經網絡（CNN），這是一種模仿人類視覺系統運作的深度學習模型。CNN 可以通過大量的標註數據進行訓練，學習圖像中的特徵，從而達到高精度的識別效果。

p.s.這個基礎教學主要環境配置是pytorch
CNN範例(影片)
https://x.com/i/status/1886134668799438996

## 你需要準備什麼
* Python 基礎語法
* 知道matplotlib怎麼用
* 一台有pytorch環境的電腦

## 類神經網路(Neural Networks)基礎
### 什麼是類神經網路，能吃嗎
類神經網路 (Neural Networks) 是一種模仿大腦神經結構的機器學習模型，在電腦視覺來說，就是模仿人體的視覺系統，例如這是一隻貓
![image](https://hackmd.io/_uploads/SJzNbdBqJx.png)
我們要怎麼判斷這是一隻貓呢 ?
那我們先來想一下貓有什麼特徵
是不是有
- 眼睛
- 耳朵
- 鬍鬚
- 毛色

但是這些特徵很多動物都有，那為甚麼我們的腦袋知道這是貓
    
#### 第一步
感知階段：光線從貓的形象反射進入眼睛，並在視網膜上形成圖像。
#### 第二步
信號傳遞：視網膜上的光感受器將光信號轉化為電信號，這些信號通過視神經傳遞到大腦。
#### 第三步
初步處理：電信號首先到達初級視覺皮層（位於大腦的枕葉），這個區域負責處理基本的視覺特徵，比如形狀、邊緣和顏色。
#### 第四步
高階處理：信息被傳遞到高階視覺皮層（顳葉和頂葉），這些區域負責整合和解釋複雜的視覺信息。
#### 第五步
物體識別：高階視覺皮層中的特定神經元對特定的形狀、紋理和顏色做出反應。通過學習和記憶，大腦能夠識別這些特徵並將它們與貓進行匹配。
#### 第六步
記憶和經驗：大腦將視覺信息與過去的經驗和知識結合起來，判斷出這是一隻貓。

![image](https://hackmd.io/_uploads/SyYf8OScJe.png)
示意圖

### bitmap
先來講一下圖片怎麼被電腦顯示出來的!
![image](https://hackmd.io/_uploads/By52yU49yl.png)
這是只有黑白圖形的時候，暗著的地方就顯示0亮著的地方就顯示1
而要顯示彩色圖片就像下面一樣把三個色彩的值用0~255表示 (~~alpha用不到不理他~~)
![image](https://hackmd.io/_uploads/r1rv1L4qJx.png)
把三個顏色的色層分開顯示就像下面這樣
![image](https://hackmd.io/_uploads/ryUTIxmqyg.png)

    
### 卷積（convolutional）
#### 在人工智慧中的應用

那來講一下在做AI的時候為甚麼需要卷積
需要捲積的理由是需要讓電腦知道這個圖片的特徵

下面是一維卷積，圖中的卷積核先經過反轉後再滑動進行輸出
![image](https://hackmd.io/_uploads/S19yw345Jx.png)!
所以算式才會變成`1x1+2x2+3x(-1)`第一個輸出才會變成2

![download](https://hackmd.io/_uploads/r1_nflQcJx.gif)
![image](https://hackmd.io/_uploads/rJBU7g791e.png)
二維卷積跟一維差不多，一樣都有卷積核，滑動，卷積操作得到特徵圖

特徵圖：所有位置的卷積結果組成特徵圖。特徵圖捕捉了圖像中具有某些特徵的區域。

### 最大池化（maxpool）
~~剛剛講完了卷積的基本原理現在來學maxpool應該不會太難吧~~
最大池化其實是卷積的一種，叫空洞卷積
目的是:
減少計算量：通過減小特徵圖的尺寸，Max pooling 可以顯著減少後續層的計算量。

降低過擬合風險：通過減少參數數量，Max pooling 可以幫助降低模型的過擬合風險。

保留重要特徵：最大池化保留了每個區域中的最大值，從而保留了圖像中最重要的特徵。

![image](https://hackmd.io/_uploads/HkHRzBaqkx.png)

### 非線性激活（non-linear）
在神經網絡中，激活函數的作用是為了引入非線性變換，從而使模型能夠學習並表示複雜的數據模式。這些函數應用於每一層神經元的輸出，賦予網絡更強的表達能力。以下是常見的非線性激活函數：

    
#### ReLU (Rectified Linear Unit)：

- 定義：$$f(x) = \max(0, x)$$
- 特點：簡單且計算高效，能夠解決梯度消失問題。
- 使用場合：廣泛應用於多層感知機和卷積神經網絡中。

#### Sigmoid (Logistic Function)：

- 定義：$$f(x) = \frac{1}{1 + e^{-x}}$$
- 特點：輸出範圍在0到1之間，有助於概率預測。
- 使用場合：通常用於二元分類問題的輸出層。

### 梯度下降(Gradient descent)
梯度下降是一種優化算法，用於最小化（或最大化）函數的值。它在機器學習和深度學習中被廣泛應用，尤其是在訓練神經網絡時，用來調整模型的參數，以降低誤差或損失。

- 初始點：從一個隨機選擇的點（或初始參數值）開始。
- 計算梯度：計算目標函數相對於所有參數的梯度。梯度是一個向量，指向函數上升最快的方向。
- 更新參數：根據梯度更新參數。更新方向與梯度方向相反，因為我們希望降低（而不是增加）目標函數的值。更新公式如下： $$\theta = \theta - \eta \cdot \nabla J(\theta)$$ 其中，$$\theta$$ 是參數，$$\eta$$ 是學習率（步長），$$\nabla J(\theta)$$ 是損失函數的梯度。
- 重複步驟：重複計算梯度和更新參數的步驟，直到算法收斂到一個最小值或達到預定的迭代次數。

### 傳播和損失函數（Forward and Loss Function） 
#### 前向傳播 (Forward Propagation)：

從輸入層開始，數據通過每一層神經元傳遞，計算出每一層的輸出，直至最終的輸出層。

計算預測結果和真實標籤之間的誤差，得到損失值（Loss）。

#### 計算梯度 (Gradient Calculation)：

從輸出層開始，計算損失函數相對於每個權重的偏導數（梯度）。這一步使用鏈式法則（Chain Rule）。

#### 反向傳播 (Backpropagation)：

將計算出的梯度從輸出層反向傳遞到輸入層，更新每一層的權重。更新公式如下： $$\theta = \theta - \eta \cdot \nabla J(\theta)$$ 其中，$$\theta$$ 是權重，$$\eta$$ 是學習率（步長），$$\nabla J(\theta)$$ 是損失函數的梯度。

#### 損失函數(loss funtion)
損失函數，也稱為成本函數（Cost Function）或目標函數（Objective Function），是機器學習和深度學習中用來衡量模型預測結果與真實標籤之間誤差的函數，簡單來說損失函數可以決定這個模型訓練得好不好。損失函數的目標是引導模型調整參數，以最小化預測誤差。以下是常見的損失函數：
    
#### 均方誤差 (Mean Squared Error, MSE)：

用於迴歸任務。

定義為預測值與真實值之間差值的平方的平均值。

計算公式： $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$ 其中，$$y_i$$ 是真實值，$$\hat{y}_i$$ 是預測值，$$n$$ 是樣本數量。

##### 交叉熵損失 (Cross-Entropy Loss)：

用於分類任務。

衡量預測的概率分佈與真實標籤的概率分佈之間的差異。

定義為所有類別上的真實標籤與預測概率的對數乘積的負和。

計算公式（對於二元分類）： $$\text{Cross-Entropy} = -\left[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})\right]$$ 其中，$$y$$ 是真實標籤，$$\hat{y}$$ 是預測概率。

### 全連接層(full_connected)
全連接層是神經網絡中的基本組成部分之一。在全連接層中，每個神經元都與上一層的所有神經元相連，這意味著每個輸入都對每個輸出有影響。全連接層通常出現在神經網絡的最後幾層，用於將學到的特徵轉換為最終的預測結果。以下是他的工作原理：

- 輸入：全連接層接收來自前一層的輸入，每個輸入都代表了一個特徵。
- 權重和偏置：每個神經元都有一組權重和一個偏置，這些參數在訓練過程中進行學習和調整。
- 線性變換：對於每個輸入，計算輸入和權重的點積，再加上偏置，進行線性變換。
- 激活函數：將線性變換的結果輸入到激活函數，以引入非線性，使模型能夠處理更複雜的數據。
- 輸出：激活函數的結果就是該層的輸出，這些輸出將傳遞到下一層。
 
## 範例程式
這個是用[cifar-10](https://www.kaggle.com/c/cifar-10/overview)的數據集做為示範的程式，利用pytorch提供的[nn.model](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)作為架構
```python!
import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib

# 定義 kohiro_model 類別
class kohiro_model(nn.Module):
    def __init__(self):
        super(kohiro_model, self).__init__()        
        self.model1 = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),  # 輸入通道3，輸出通道32，卷積核尺寸5×5，步長1，填充2    
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),  # 展平後變成 64*4*4 了
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )
        
    def forward(self, x):
        x = self.model1(x)
        return x

# 準備數據集
train_data = torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)       
test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)       

# length 長度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10，則打印：訓練數據集的長度為：10
print("訓練數據集的長度：{}".format(train_data_size))
print("測試數據集的長度：{}".format(test_data_size))

# 利用 Dataloader 來加載數據集
train_dataloader = DataLoader(train_data, batch_size=64)        
test_dataloader = DataLoader(test_data, batch_size=64)

# 創建網絡模型
model = kohiro_model() 
if torch.cuda.is_available():
    model = model.cuda() # 網絡模型轉移到cuda上

# 損失函數
loss_fn = nn.CrossEntropyLoss() # 交叉熵，fn 是 function 的縮寫
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()        # 損失函數轉移到cuda上

# 優化器
learning = 0.01  # 1e-2 就是 0.01 的意思
optimizer = torch.optim.SGD(model.parameters(),learning)   # 隨機梯度下降優化器  

# 設置網絡的一些參數
# 記錄訓練的次數
total_train_step = 0
# 記錄測試的次數
total_test_step = 0

# 訓練的輪次
epoch = 10

# 添加 tensorboard
writer = SummaryWriter("logs")

for i in range(epoch):
    print("-----第 {} 輪訓練開始-----".format(i+1))
    
    # 訓練步驟開始
    model.train() # 當網絡中有dropout層、batchnorm層時，這些層能起作用
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()  # 數據放到cuda上
            targets = targets.cuda() # 數據放到cuda上
        outputs = model(imgs)
        loss = loss_fn(outputs, targets) # 計算實際輸出與目標輸出的差距
        
        # 優化器對模型調優
        optimizer.zero_grad()  # 梯度清零
        loss.backward() # 反向傳播，計算損失函數的梯度
        optimizer.step()   # 根據梯度，對網絡的參數進行調優
        
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("訓練次數：{}，Loss：{}".format(total_train_step,loss.item()))  # 方式二：獲得loss值
            writer.add_scalar("train_loss",loss.item(),total_train_step)
    
    # 測試步驟開始（每一輪訓練後都查看在測試數據集上的loss情況）
    model.eval()  # 當網絡中有dropout層、batchnorm層時，這些層不能起作用
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 沒有梯度了
        for data in test_dataloader: # 測試數據集提取數據
            imgs, targets = data # 數據放到cuda上
            if torch.cuda.is_available():
                imgs = imgs.cuda() # 數據放到cuda上
                targets = targets.cuda()
            outputs = model(imgs)
            loss = loss_fn(outputs, targets) # 僅data數據在網絡模型上的損失
            total_test_loss = total_test_loss + loss.item() # 所有loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
            
    print("整體測試集上的Loss：{}".format(total_test_loss))
    print("整體測試集上的正確率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)  
    total_test_step = total_test_step + 1
    
    torch.save(model, "./model/kohiro_model_{}.pth".format(i)) # 保存每一輪訓練後的結果
    torch.save(model.state_dict(),"kohiro_model_{}.path".format(i)) # 保存方式二         
    print("模型已保存")
    
writer.close()


```

