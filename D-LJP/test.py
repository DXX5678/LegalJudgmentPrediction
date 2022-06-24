import torch
from torch import optim
from pytorch_pretrained_bert.tokenization import BertTokenizer
from model.MergeModel import MergeModel
from tool.test_tool import test

if __name__ == "__main__":
    """
    fact = "上海市徐汇区人民检察院指控：2011年至2014年间，被告人罗俊先后利用担任上海市公安局交通警察总队高架支队审理办案科承办人、事故组组长的职务便利，收受、索取当事人贿赂共计人民币17万元，分述如下：1、2011" \
           "年11月，被告人罗俊在处理当事人张某某与他人发生的交通事故时，接受事故代理人施某的请托，认定事故双方同等责任，后于2014年，罗俊收受张某某通过施某给予的感谢费人民币2万元。2、2013" \
           "年，被告人罗俊在违法当事人李某某通过张某提出从宽处罚的请托后，通过张某向李某某索要人民币5万元。3、2014年4" \
           "月，被告人罗俊在参与赵某危险驾驶案现场勘查处置后，伙同其妻勇某（另案处理）接受赵某的请托，在案件处理过程中为赵某提供便利，并由勇某收受赵某给予的人民币10万元。 " \
           "2014年7月29日下午，被告人罗俊向上海市长宁区人民检察院投案，并供述以上基本事实。罗俊在家属帮助下退还赃款人民币13万元。 "
    """
    fact = input("案由陈述：")
    model_path = "model/result/17.pkl"
    bert_tokenizer_path = "bert-chinese/vocab.txt"
    bert_path = "bert-chinese/"
    batch_size = 32
    learning_rate = 1e-4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state = torch.load(model_path)
    model = MergeModel(bert_path, batch_size, device).to(device)
    model.load_state_dict(state['net'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    optimizer.load_state_dict(state['optimizer'])
    tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
    zm, ft = test(model, tokenizer, fact, device)
    f = open('content/article_content_dict.pkl', 'rb')
    content = torch.load(f)
    print("指控预测：犯" + "、".join(zm))
    print("法条预测：(违反下列刑法条例)")
    for i in ft:
        print(i, content[i], sep='：')
