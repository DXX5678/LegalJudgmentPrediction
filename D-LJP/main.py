# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import torch


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':

    print_hi("pycharm")
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    a = torch.Tensor([[1, 2]])
    temp = a
    for i in range(31):
        a = torch.cat((a, temp), dim=0)
    a = a.cuda()
    print(a[0, 0])
    print(a.shape)

    # f = open('content/article_content_dict.pkl', 'rb')
    # data = torch.load(f)
    # print(data)
    # print(len(data))
