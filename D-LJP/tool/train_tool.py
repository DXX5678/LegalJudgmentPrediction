import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from timeit import default_timer as timer
import time
from tool.eval_tool import valid
from tool.save_tool import save_checkpoint


def train(train_dataloader, eval_dataloader, test_dataloader, model, optimizer, epochs, eval_per_epoch, output_path, log_path):
    print("训练开始啦！！！")
    data_len = len(train_dataloader)
    for epoch in range(epochs):
        start_time = timer()
        current_epoch = epoch
        print("第"+str(current_epoch + 1)+"次迭代开始：", time.ctime())
        model.train()
        average_loss = 0.0
        total_loss = 0.0
        losses = []
        for batch_idx, data in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            result = model(data)
            loss, acc_result = result["loss"], result["acc_result"]
            total_loss += loss.item()
            if batch_idx == 0:
                losses.append(loss.item())
            elif (batch_idx+1) % 50 == 0:
                losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if batch_idx == data_len - 1:
                average_loss = total_loss / (batch_idx + 1)
        delta_t = timer() - start_time
        print("第"+str(current_epoch + 1)+"次迭代结束，历时：{}，平均损失值：{}".format(delta_t, average_loss))
        plt.figure()
        plt.plot(losses)
        plt.savefig(log_path + str(time.time()) + ".png")
        save_checkpoint(os.path.join(output_path, "%d.pkl" % current_epoch), model, optimizer, epoch)
        if eval_per_epoch != 0 and (current_epoch + 1) % eval_per_epoch == 0:
            print("验证集：")
            valid(eval_dataloader, model, current_epoch, log_path)
            print("测试集：")
            valid(test_dataloader, model, current_epoch, log_path)
        if eval_per_epoch == 0:
            if current_epoch == (epochs - 1):
                print("验证集：")
                valid(eval_dataloader, model, current_epoch, log_path)
                print("测试集：")
                valid(test_dataloader, model, current_epoch, log_path)