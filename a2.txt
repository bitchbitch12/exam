import numpy as np
import random
import collections

def extract_elements(lst, i=1):
  return [items[i] for items in lst]

def static_ranking(n,m):
    df_train=np.random.randint(1,6, size=(n,m))
    print("df_train: ",df_train)
    ranks_sum=list(df_train.sum(axis=0))
    print("Ranks_sum: ",ranks_sum)
    ranks_avg=[[(i/n).round(2)] for i in ranks_sum]
    print("Ranks_avg: ",ranks_avg)
    for i in range(len(ranks_avg)):
      ranks_avg[i].append(ranks_avg.index(ranks_avg[i]))
      print(i, ranks_avg)
    offline_ranks=sorted(ranks_avg,key=lambda l:l[0], reverse=True) 
    print("Offline_Ranks: ",offline_ranks)
    return extract_elements(offline_ranks,1)

def MAB(offline_ranking):
    item_pref=[]
    for j in range(10):
      item_pref.append(random.choice(offline_ranking))
    return collections.Counter(item_pref)

def final_ranking(presentation_ranking):
    sorted_dict={k: v for k, v in sorted(presentation_ranking.items(), key=lambda item: item[1],reverse=True)}
    print('Final ranking of items:', list(sorted_dict.keys()))

if __name__ == "__main__":
    n=int(input("Enter the number of users:"))
    m=int(input("Enter the number of items:"))
    offline_ranking=static_ranking(n,m)
    print(offline_ranking)
    presentation_ranking={}
    k=int(input('Enter the value of k:'))
    for i in range(len(offline_ranking[0:k])):
      print("i:",i)
      if(i==0):
        for val in offline_ranking[0:k]:
          presentation_ranking[val]=0
      dup=MAB(offline_ranking[0:k])
      print("Dup:",dup)
      for j in offline_ranking[0:k]:
        try:
          presentation_ranking[j]+=dup[j]
        except:
          presentation_ranking[j]=presentation_ranking[j]
    print('Offline ranking of items:',offline_ranking)
    final_ranking(presentation_ranking)