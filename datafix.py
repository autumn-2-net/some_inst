import json
# p ='d' in 'abc'
# print(p)
#
#
# with open('stroke-order-jian.json','r',encoding='utf-8') as f:
#     a =f.read()
#     aj =json.loads(a)
# with open('stroke-table.json','r',encoding='utf-8') as f:
#     b =f.read()
#     bj =json.loads(b)
# ccccc ={}
# for i in aj:
#     list1 =''
#     cc =aj[i]
#     if 'x' in cc:
#         continue
#     if 'w' in cc:
#         continue
#     if 'y' in cc:
#         continue
#     if 'e' in cc:
#         continue
#     if 'v' in cc:
#         continue
#     for ii in cc:
#         acac=bj[ii]['shape']
#         list1=acac+list1
#     ccccc[i]=list1
# print(ccccc)
# dddd =json.dumps(ccccc,ensure_ascii=False)
# with open('fix1.json','w',encoding='utf-8') as wf:d
#     wf.write(dddd)

with open('fix1.json','r',encoding='utf-8') as f:
    a =f.read()

with open('映射.json','r',encoding='utf-8') as f:
    b =f.read()
ass =json.loads(a)
abss =json.loads(b)
jsonnn ={}
for i in ass:
    list1=[]
    cc =ass[i]
    for j in cc:
        asda =abss.get(j)
        list1.append(asda)
    jsonnn[i]=list1

print(len(jsonnn))


