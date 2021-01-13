import os
import pickle
import sys

import torch


def main():
    with open('_allData.p', 'rb') as meta:
        u2e, i2e, u_train, i_train, r_train, u_test, i_test, r_test, u_adj, i_adj = pickle.load(meta)

    userlist = []
    u_neighs = {}
    itemlist = []
    i_neighs = {}
    for user in u_train:

        if user not in userlist:
            # print("当前用户：", user)
            uList = []
            ulist = {}  # 邻居
            userlist.append(user)
            item = [it for it in u_adj[user]]
            # print("购买过的商品数量：", item.__len__(),item)
            for it in item:
                lists = [li for li in i_adj[it[0]] if li[0] != user and li[1] == it[1]]

                it = it[0]
                # print(it, "还有", len(lists), "人买", lists)
                lists = [li[0] for li in lists]
                for kk in lists:
                    # kk : 购买过同一件物品的用户 ulist 件数
                    if kk not in ulist.keys():
                        ulist[kk] = 1
                    else:
                        ulist[kk] = ulist[kk] + 1

            count = 0
            for key in ulist.keys():
                count += ulist.get(key)
            avg = count / len(ulist)

            # print("user 的邻居总共有：", len(ulist), "个")
            for neu in ulist.keys():
                if ulist.get(neu) > avg:
                    uList.append(neu)
                    # print(neu, "购买", ulist.get(neu), "件")
            # print("邻居平均购买件数为：", avg)
            # print("大于平均购买件数的邻居有：", len(uList), "个")

            # break

            # print(user, " | ", ulist.__len__())

            u_neighs[user] = uList

    for item in i_train:

        if item not in itemlist:
            # print("当前商品：", item)
            iList = []
            ilist = {}  # 邻居
            itemlist.append(item)
            user = [it for it in i_adj[item]]
            # print("购买过该商品的用户数量：", user.__len__(), user)
            for us in user:
                lists = [li for li in u_adj[us[0]] if li[0] != item and li[1] == us[1]]
                if len(lists) == 0:
                    lists = [li for li in u_adj[us[0]] if li[0] != item]

                us = us[0]
                # print(us, "还买了", len(lists), "件：", lists)
                lists = [li[0] for li in lists]
                for kk in lists:
                    # kk : 购买过同一件物品的用户 ulist 件数
                    if kk not in ilist.keys():
                        ilist[kk] = 1
                    else:
                        ilist[kk] = ilist[kk] + 1

            count = 0
            for key in ilist.keys():
                count += ilist.get(key)

            avg = count / len(ilist)

            # print("item 的邻居总共有：", len(ilist), "个")
            for neu in ilist.keys():
                if ilist.get(neu) > avg:
                    iList.append(neu)
                    # print(neu, "购买", ilist.get(neu), "件")
            # print("邻居平均购买件数为：", avg)
            # print("大于平均购买件数的邻居有：", len(iList), "个")
            #
            # break

            # print(user, " | ", ulist.__len__())

            i_neighs[item] = iList
    # print(imap.keys().__len__())
    with open('yelp.p', 'wb') as meta:
        pickle.dump((u_neighs, i_neighs), meta)


if __name__ == '__main__':
    main()
