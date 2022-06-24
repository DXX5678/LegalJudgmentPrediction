import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from timeit import default_timer as timer
import time
from tool.accuracy_tool import gen_micro_macro_result


def valid(eval_dataloader, model, current_epoch, log_path):
    print("评价开始啦！！！")
    data_len = len(eval_dataloader)
    model.eval()
    total_loss = 0.0
    ft_total_mif = 0.0
    ft_total_maf = 0.0
    ft_total_mip = 0.0
    ft_total_map = 0.0
    ft_total_mir = 0.0
    ft_total_mar = 0.0
    ft_acc_mip = []
    ft_acc_mir = []
    ft_acc_mif = []
    ft_acc_map = []
    ft_acc_mar = []
    ft_acc_maf = []
    zm_total_mif = 0.0
    zm_total_maf = 0.0
    zm_total_mip = 0.0
    zm_total_map = 0.0
    zm_total_mir = 0.0
    zm_total_mar = 0.0
    zm_acc_mip = []
    zm_acc_mir = []
    zm_acc_mif = []
    zm_acc_map = []
    zm_acc_mar = []
    zm_acc_maf = []
    start_time = timer()
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(eval_dataloader)):
            result = model(data)
            loss, acc_result = result["loss"], result["acc_result"]
            total_loss += loss.item()
            ft_eval_results = gen_micro_macro_result(acc_result["ft"])
            zm_eval_results = gen_micro_macro_result(acc_result["zm"])
            ft_total_mif += ft_eval_results["mif"]
            ft_total_maf += ft_eval_results["maf"]
            zm_total_mif += zm_eval_results["mif"]
            zm_total_maf += zm_eval_results["maf"]
            ft_total_mip += ft_eval_results["mip"]
            ft_total_map += ft_eval_results["map"]
            zm_total_mip += zm_eval_results["mip"]
            zm_total_map += zm_eval_results["map"]
            ft_total_mir += ft_eval_results["mir"]
            ft_total_mar += ft_eval_results["mar"]
            zm_total_mir += zm_eval_results["mir"]
            zm_total_mar += zm_eval_results["mar"]
            if batch_idx == 0:
                ft_acc_mip.append(ft_eval_results["mip"])
                ft_acc_mir.append(ft_eval_results["mir"])
                ft_acc_mif.append(ft_eval_results["mif"])
                ft_acc_map.append(ft_eval_results["map"])
                ft_acc_mar.append(ft_eval_results["mar"])
                ft_acc_maf.append(ft_eval_results["maf"])
                zm_acc_mip.append(zm_eval_results["mip"])
                zm_acc_mir.append(zm_eval_results["mir"])
                zm_acc_mif.append(zm_eval_results["mif"])
                zm_acc_map.append(zm_eval_results["map"])
                zm_acc_mar.append(zm_eval_results["mar"])
                zm_acc_maf.append(zm_eval_results["maf"])
            elif (batch_idx + 1) % 5 == 0:
                ft_acc_mip.append(ft_eval_results["mip"])
                ft_acc_mir.append(ft_eval_results["mir"])
                ft_acc_mif.append(ft_eval_results["mif"])
                ft_acc_map.append(ft_eval_results["map"])
                ft_acc_mar.append(ft_eval_results["mar"])
                ft_acc_maf.append(ft_eval_results["maf"])
                zm_acc_mip.append(zm_eval_results["mip"])
                zm_acc_mir.append(zm_eval_results["mir"])
                zm_acc_mif.append(zm_eval_results["mif"])
                zm_acc_map.append(zm_eval_results["map"])
                zm_acc_mar.append(zm_eval_results["mar"])
                zm_acc_maf.append(zm_eval_results["maf"])
    average_loss = total_loss / data_len
    average_ft_mif = ft_total_mif / data_len
    average_ft_maf = ft_total_maf / data_len
    average_zm_mif = zm_total_mif / data_len
    average_zm_maf = zm_total_maf / data_len
    average_ft_mip = ft_total_mip / data_len
    average_ft_map = ft_total_map / data_len
    average_zm_mip = zm_total_mip / data_len
    average_zm_map = zm_total_map / data_len
    average_ft_mir = ft_total_mir / data_len
    average_ft_mar = ft_total_mar / data_len
    average_zm_mir = zm_total_mir / data_len
    average_zm_mar = zm_total_mar / data_len
    delta_t = timer() - start_time
    print("第" + str(current_epoch + 1) + "次迭代后，评价结束，历时：{}，平均损失值：{}".format(delta_t, average_loss))
    title1 = "FT eval: MIP " + str(average_ft_mip) + " MAP " + str(average_ft_map) + " MIR " + str(
        average_ft_mir) + " MAR " + str(average_ft_mar) + " MIF " + str(average_ft_mif) + " MAF " + str(average_ft_maf)
    title2 = "ZM eval: MIP " + str(average_zm_mip) + " MAP " + str(average_zm_map) + " MIR " + str(
        average_zm_mir) + " MAR " + str(average_zm_mar) + " MIF " + str(average_zm_mif) + " MAF " + str(average_zm_maf)
    print(title1)
    print(title2)
    fig1 = plt.figure(num=title1)
    ax11 = fig1.add_subplot(231)
    ax11.set_title("MIP")
    ax11.plot(ft_acc_mip)
    ax12 = fig1.add_subplot(232)
    ax12.set_title("MIR")
    ax12.plot(ft_acc_mir)
    ax13 = fig1.add_subplot(233)
    ax13.set_title("MIF")
    ax13.plot(ft_acc_mif)
    ax14 = fig1.add_subplot(234)
    ax14.set_title("MAP")
    ax14.plot(ft_acc_map)
    ax15 = fig1.add_subplot(235)
    ax15.set_title("MAR")
    ax15.plot(ft_acc_mar)
    ax16 = fig1.add_subplot(236)
    ax16.set_title("MAF")
    ax16.plot(ft_acc_maf)
    fig1.savefig(log_path + "eval/" + str(time.time()) + "ft.png")
    fig2 = plt.figure(num=title2)
    ax21 = fig2.add_subplot(231)
    ax21.set_title("MIP")
    ax21.plot(zm_acc_mip)
    ax22 = fig2.add_subplot(232)
    ax22.set_title("MIR")
    ax22.plot(zm_acc_mir)
    ax23 = fig2.add_subplot(233)
    ax23.set_title("MIF")
    ax23.plot(zm_acc_mif)
    ax24 = fig2.add_subplot(234)
    ax24.set_title("MAP")
    ax24.plot(zm_acc_map)
    ax25 = fig2.add_subplot(235)
    ax25.set_title("MAR")
    ax25.plot(zm_acc_mar)
    ax26 = fig2.add_subplot(236)
    ax26.set_title("MAF")
    ax26.plot(zm_acc_maf)
    fig2.savefig(log_path + "eval/" + str(time.time()) + "zm.png")
