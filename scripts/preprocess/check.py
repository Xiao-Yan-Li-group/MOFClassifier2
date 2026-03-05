import json


# data = json.load(open("../../dataset/label/manual_Tom.json"))

# i=0
# j=0
# k=0
# for cif_id in data:
#     if data[cif_id]["has_error"] == 0:
#         i+=1
#     elif data[cif_id]["has_error"] == 1:
#         j+=1
#     k+=1
# print(i, j, k)


data1 = json.load(open("../../ft-llama/data/mofclass2_robocry_reason_train.json"))
data2 = json.load(open("../../ft-llama/data/mofclass2_robocry_reason_test.json"))
print(len(data1), len(data2), len(data1)+len(data2))

data1 = json.load(open("../../ft-llama/data/mofclass2_mof2text_reason_train.json"))
data2 = json.load(open("../../ft-llama/data/mofclass2_mof2text_reason_test.json"))
print(len(data1), len(data2), len(data1)+len(data2))
