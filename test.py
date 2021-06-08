# 程序中的datatrain.txt和datatest.txt中，要去掉rensor，否则读取文件时会报错
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
from capture import getdata
root = "./image"
LR = 0.001
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        fh = open(txt, 'r')  # 以只读的方式打开txt文件
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split(',')  # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            imgs.append((words[0], int(words[1])))  # #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # 根据两个txt的内容，words[0]是图片位置信息，words[1]是图片标签信息
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader =lambda x:(cv.resize(cv.imread(x),(28,28), interpolation=cv.INTER_LINEAR))[:,:,::-1].copy().transpose((2,0,1))


    def __getitem__(self, index):  # 按照索引方式读取
        fn, label = self.imgs[index]  # fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = self.loader(fn)  # 将图片转为RGB格式
        print(img.shape,type(img))
        if self.transform is not None:
            img = self.transform(img)  # 转换图片格式，由第41行代码的transform可知将图片转为Tensor格式
        return img, label

    def __len__(self):
        return len(self.imgs)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 5, 1, 2),  # 最后一个参数要满足=（第二个参数-1）/2，这样默认图片的长宽不变，只是改变通道数
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(1568, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
        )

    def forward(self, x):
        conv1_out = self.conv1(x.permute(0,2,1,3))
        conv2_out = self.conv2(conv1_out)
        print(conv2_out.shape)
        res = conv2_out.view(conv2_out.size(0), -1)
        print(res.shape)
        out = self.dense(res)
        return out


def train():
    train_data = MyDataset(txt="test.txt", transform=transforms.ToTensor())
    # 程序中直接用opencv调用测试图像，因此不需要再单独加载测试数据
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(1000):
        print('epoch {}'.format(epoch + 1))
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            batch_y-=1
            loss = loss_func(out, torch.max(batch_y.long(),torch.Tensor([0]).long()))
            print(float(loss.item()))
            optimizer.zero_grad()
            loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), 'model.pkl')  # 保存模型
def test():
    model = Net()
    model.load_state_dict(torch.load('model.pkl'))  # 加载模型
    model.eval()
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 64), cap.set(cv.CAP_PROP_FRAME_HEIGHT, 48)
    while 1:
        ret, frame = cap.read()
        gray = cv.resize(cv.cvtColor(frame, cv.COLOR_BGR2GRAY),(28,28), interpolation=cv.INTER_LINEAR)
        print(gray.shape)
        img_to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ]
        )  # 将opencv读取的图像转为tensor格式
        tensor = img_to_tensor(gray)
        # print(tensor.shape)  # 数据大小为【3，28，28】
        inputs = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)  # 改变数据维度
        # print(inputs.shape)  # 数据大小为【1，3，28，28】
        test_output = model(inputs.permute(0,2,1,3))
        print(test_output)
        pred_y = torch.max(test_output, 1)[1].data.item()+1
        print(pred_y, 'prediction number')
        cv.imshow("frame", gray)
        if cv.waitKey(0)==ord('q'):
            break
        else:
            continue
    cv.destroyAllWindows()

if __name__=="__main__":
    train()