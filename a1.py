import random
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import statistics

arm_reward = []
#n = 10
n = int(input("Enter the number of iterations: "))

'''for i in range(0, n):
  arm = random.randint(1, 5) 
  reward = random.randint(1, 10)
  arm_reward.append((arm,reward))'''
arm = []
for i in range(0, n):
  arm.append(random.randint(1, 5))
#a_list = np.random.randn(n)
r_list = np.random.randn(n)
#arm = [abs(round(i*10)) for i in a_list]
reward = [abs(round(i*10)) for i in r_list]
for i in range(0, n):
  arm_reward.append((arm[i], reward[i]))

print(arm_reward)

"""## 1.Greedy approach"""

def best_arm_greedy(arm_reward):
  a_11 = []  #average 
  for i in range(1, 6): #arm - 1 to 5
    sum1 = 0
    cnt = 0
    for j in range(len(arm_reward)):
      if arm_reward[j][0] == i:
        sum1 += arm_reward[j][1]
        cnt += 1
    if cnt > 0:
      a_11.append(sum1/cnt) 
    else:
      a_11.append(0)
  max_value = max(a_11)
  max_index = a_11.index(max_value)
  return max_index + 1#,max_value, a_11

print("The best arm to be chosen for the 11th trail is: ",best_arm_greedy(arm_reward))


"""## 2.Incremental Uniform"""

def best_arm_ic(arm_reward):
  a_11 = []
  for i in range(1, 6): #arm - 1 to 5
    qn = 0
    cnt = 1
    for j in range(0, n):
      if arm_reward[j][0] == i:
        qn1 = qn + ((arm_reward[j][1] - qn)/cnt)
        qn = qn1
        cnt += 1
    a_11.append(qn)
  max_value = max(a_11)
  max_index = a_11.index(max_value)
  return max_index + 1#, max_value, a_11

print("The best arm to be chosen for the 11th trail is: ",best_arm_ic(arm_reward))

"""## 3.Epsilon Greedy"""

epsilon = 0.3
num_iter = 30
exp_reward = [0.0 for i in range(0, 5)] #initially assign 0.0 for 5 arms -> qn
cnt_arm = [0 for i in range(0, 5)]

def choose_arm(p):
  if p > epsilon:
    #greedy choice
    return exp_reward.index(max(exp_reward)) + 1, "greedy"
  else:
    #random choice
    return random.randint(1, 5), "random"

print("Initial Expected Reward: ", exp_reward)
print("Initial count of each arm: ", cnt_arm)
print("************************************")
reward_total = []
for i in range(0, num_iter):
  prob = random.uniform(0, 1)
  arm_no, greedy_random = choose_arm(prob)
  temp_reward = abs(round(np.random.randn(1)[0]*10))
  cnt_arm[arm_no - 1] += 1
  new_value = exp_reward[arm_no - 1] + ((temp_reward - exp_reward[arm_no - 1])/cnt_arm[arm_no - 1]) #qn1 = qn+(1/n)*(rn - qn)
  exp_reward[arm_no - 1] = new_value #qn1=qn
  reward_total.append(sum(exp_reward)/len(exp_reward)) #for graph - average exp_reward of each epsilon
  print("Greedy/Random: ", greedy_random)
  print("Iteration ", i)
  print("Arm no: ", arm_no)
  print("Reward of each arm: ", temp_reward)
  print("Expected reward: ", exp_reward)
  print("Count of each arm: ", cnt_arm)

iter = [i for i in range(1, 31)]
sns.lineplot(x = reward_total, y = iter)

"""## 4.Epsilon Greedy Graph"""

epsilon_1 = 0.0
epsilon_2 = 0.1
epsilon_3 = 0.3
num_iter = 500
exp_reward_1 = [0.0 for i in range(0, 5)]
cnt_arm_1 = [0 for i in range(0, 5)]
exp_reward_2 = [0.0 for i in range(0, 5)]
cnt_arm_2 = [0 for i in range(0, 5)]
exp_reward_3 = [0.0 for i in range(0, 5)]
cnt_arm_3 = [0 for i in range(0, 5)]

def choose_arm(p, epsilon, exp_reward):
  if p > epsilon:
    #greedy choice
    return exp_reward.index(max(exp_reward)) + 1, "greedy"
  else:
    #random choice
    return random.randint(1, 5), "random"

# 1)
print("Initial Expected Reward: ", exp_reward_1)
print("Initial count of each arm: ", cnt_arm_1)
print("************************************")
reward_total_1 = []
for i in range(0, num_iter):
  prob = random.uniform(0, 1)
  arm_no, greedy_random = choose_arm(prob, epsilon_1, exp_reward_1)
  temp_reward = abs(round(np.random.randn(1)[0]*10))
  cnt_arm_1[arm_no - 1] += 1
  new_value = exp_reward_1[arm_no - 1] + ((temp_reward - exp_reward_1[arm_no - 1])/cnt_arm_1[arm_no - 1])
  exp_reward_1[arm_no - 1] = new_value
  reward_total_1.append(sum(exp_reward_1)/len(exp_reward_1))
  print("Greedy/Random: ", greedy_random)
  print("Iteration ", i)
  print("Arm no: ", arm_no)
  print("Reward of each arm: ", temp_reward)
  print("Expected reward: ", exp_reward_1)
  print("Count of each arm: ", cnt_arm_1)

# 2)
print("Initial Expected Reward: ", exp_reward_2)
print("Initial count of each arm: ", cnt_arm_2)
print("************************************")
reward_total_2 = []
for i in range(0, num_iter):
  prob = random.uniform(0, 1)
  arm_no, greedy_random = choose_arm(prob, epsilon_2, exp_reward_2)
  temp_reward = abs(round(np.random.randn(1)[0]*10))
  cnt_arm_2[arm_no - 1] += 1
  new_value = exp_reward_2[arm_no - 1] + ((temp_reward - exp_reward_2[arm_no - 1])/cnt_arm_2[arm_no - 1])
  exp_reward_2[arm_no - 1] = new_value
  reward_total_2.append(sum(exp_reward_2)/len(exp_reward_2))
  print("Greedy/Random: ", greedy_random)
  print("Iteration ", i)
  print("Arm no: ", arm_no)
  print("Reward of each arm: ", temp_reward)
  print("Expected reward: ", exp_reward_2)
  print("Count of each arm: ", cnt_arm_2)

# 3)
print("Initial Expected Reward: ", exp_reward_3)
print("Initial count of each arm: ", cnt_arm_3)
print("************************************")
reward_total_3 = []
for i in range(0, num_iter):
  prob = random.uniform(0, 1)
  arm_no, greedy_random = choose_arm(prob, epsilon_3, exp_reward_3)
  temp_reward = abs(round(np.random.randn(1)[0]*10))
  cnt_arm_3[arm_no - 1] += 1
  new_value = exp_reward_3[arm_no - 1] + ((temp_reward - exp_reward_3[arm_no - 1])/cnt_arm_3[arm_no - 1])
  exp_reward_3[arm_no - 1] = new_value
  reward_total_3.append(sum(exp_reward_3)/len(exp_reward_3))
  print("Greedy/Random: ", greedy_random)
  print("Iteration ", i)
  print("Arm no: ", arm_no)
  print("Reward of each arm: ", temp_reward)
  print("Expected reward: ", exp_reward_3)
  print("Count of each arm: ", cnt_arm_3)

iter = [i for i in range(1, 501)]  

plt.figure(figsize=(12, 5))
# plot lines
plt.plot(iter, reward_total_1, label = "epsilon = 0.0")
plt.plot(iter, reward_total_2, label = "epsilon = 0.1")
plt.plot(iter, reward_total_3, label = "epsilon = 0.3")
plt.xlabel('Number of iterations')
plt.ylabel('Average Reward')
plt.legend()
plt.show()

"""## 5.Upper Confidence Bound (UCB)"""

def best_arm_ucb(arm_reward):
  a_11 = []
  cnt_arm = []
  for i in range(1, 6): #arm - 1 to 5
    qn = 0
    cnt = 1
    for j in range(0, n):
      if arm_reward[j][0] == i:
        qn1 = qn + ((arm_reward[j][1] - qn)/cnt)
        qn = qn1
        cnt += 1
    a_11.append(qn)    #will have expected reward for each arm
    cnt_arm.append(cnt)
  return a_11, cnt_arm

def ucb_algo(arm_reward, c):
  qt_a, cnt = best_arm_ucb(arm_reward)
  final_reward = qt_a
  for i in range(len(final_reward)):
    final_reward[i] += c*(math.sqrt(math.log(sum(cnt)/cnt[i])))
  return final_reward

c = 1
temp_list = ucb_algo(arm_reward, c)
print("Best arm to be chosen in the 11th iteration using UCB algorithm: ",temp_list.index(max(temp_list)) + 1)

"""## 6.Naive PAC"""

epsilon = float(input("Enter the value of epsilon: "))
delta = float(input("Enter the value of delta: "))
arm_n = int(input("Enter the number of arms: "))

def pac_naive(epsilon, delta, arm_n):
  l = round((4/(epsilon*epsilon))*math.log((2*arm_n)/delta))
  arm_reward = []
  for i in range(0, arm_n):
    r_list = np.random.randn(l)
    reward = [abs(round(i*10)) for i in r_list]
    arm_reward.append(sum(r_list)/len(r_list))
  return arm_reward

list1 = pac_naive(epsilon, delta, arm_n)
print("The best arm for the next iteration ({}% confidence) using Naive PAC is: ".format((1-delta)*100), list1.index(max(list1)) + 1)

"""## 7.Median elimination PAC"""

epsilon_m = float(input("Enter the value of epsilon: "))
delta_m = float(input("Enter the value of delta: "))
arm_n_m = int(input("Enter the number of arms: "))

def me_pac(e, d, a):
  cnt = 1
  arm_reward = {}
  for i in range(1, 65):
    arm_reward[i] = 0.0
  l = 1   
  while(len(arm_reward) != 1):
    print("********************************")
    print("Iteration {}: ".format(cnt))
    arm_reward_new = {}
    if l == 1:
      e = e/4
      d = d/2
    else:
      e = (3*e)/4
      d = d/2
    iter = round(1/((e*e)/2)*math.log(3/d))
    print("Epsilon: ", e)
    print("Delta: ", d)
    print("Number of iterations: ", iter)    #no.of samples
    # print("Length of arm reward: ", len(arm_reward))
    # print("Length of arm reward new: ", len(arm_reward_new))
    # for i in range(len(arm_reward)):
    for key, value in arm_reward.items():
      r_list = np.random.randn(iter)
      reward = [abs(round(i*10)) for i in r_list]
      # print(reward)
      arm_reward[key] = (arm_reward[key] + (sum(reward)/len(reward)))/2
    # print("Arm reward before sorted: ", arm_reward)
    arm_reward = dict(sorted(arm_reward.items(), key=lambda x:x[1]))
    # print("Arm reward after sorted: ", arm_reward)
    # print("Length of sorted arm: ",len(arm_reward))
    median = statistics.median(list(arm_reward.values()))
    print("Median: ", median)
    for i in arm_reward:
      if arm_reward[i] > median:
        arm_reward_new[i] = arm_reward[i]
    # print("Length of new arm reward: ", len(arm_reward_new))
    arm_reward = arm_reward_new
    cnt += 1
    l += 1
  return list(arm_reward.keys())[0]

print("The best arm for the next iteration ({}% confidence) using Median Elimination PAC is: ".format((1-delta_m)*100), me_pac(epsilon_m, delta_m, arm_n_m))