import re
import torch
from LJPapp.tool.prediction_tool import predicate_process


def test(model, tokenizer, fact, device):
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，、（）“”]+'
    fact = fact.replace('\n', '').replace('\r', '').replace('\t', '').replace(' ', '')
    fact = re.sub(r, '', fact)
    fact = fact.strip()
    input_data = tokenizer.tokenize(fact)
    if len(input_data) <= 510:
        input_data = ["[CLS]"] + input_data + ["[SEP]"]
        while len(input_data) < 512:
            input_data.append("[PAD]")
    else:
        input_data = input_data[0:510]
        input_data = ["[CLS]"] + input_data + ["[SEP]"]
    input_data = torch.tensor([tokenizer.convert_tokens_to_ids(input_data)])
    temp = input_data
    for i in range(31):
        input_data = torch.cat((input_data, temp), dim=0)
    input_data = input_data.to(device)
    model.eval()
    with torch.no_grad():
        result = model([input_data, None, None])
    prediction_zm = result["output"]["zm"]
    prediction_ft = result["output"]["ft"]
    output_zm = predicate_process(prediction_zm, "zm")
    output_ft = predicate_process(prediction_ft, "ft")
    return output_zm, output_ft
