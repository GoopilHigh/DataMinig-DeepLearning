import numpy as np
from sklearn.datasets import load_boston
from sklearn import linear_model
import matplotlib.pyplot as plt
import random



# parameter definition
split_coef=0.5
print_able=1
draw_able=1
gaussian_noise_able=0
gaussian_noise_sigma=1


# In[3]:


#intialization
lr=linear_model.LinearRegression()
boston=load_boston()
if gaussian_noise_able:
    noise_map = np.random.normal(0,gaussian_noise_sigma,boston.data.shape[0]*boston.data.shape[1])
    noise_map = noise_map.reshape(boston.data.shape[0],boston.data.shape[1])
    boston.data = boston.data + noise_map


# In[4]:


# random for samples
index=[i for i in range(len(boston.data))]
random.shuffle(index)
data=boston.data[index]
target=boston.target[index]


# In[5]:


# divide samples to training samples and test samples by threshold
threshold=round(split_coef*len(data))
boston_x_train=data[:threshold]
boston_x_test=data[threshold:]
boston_y_train=target[:threshold]
boston_y_test=target[threshold:]


# In[6]:


# predict target with each of other variable
each_score=np.zeros(len(data[0]))
each_trainError=np.zeros(len(data[0]))
each_testError=np.zeros(len(data[0]))
for i in range(len(data[0])):
    # construct model via each variable
    lr.fit(boston_x_train[:,i:i+1],boston_y_train)
    # calculate score of each model
    each_score[i]=lr.score(boston_x_test[:,i:i+1],boston_y_test)
    # calculate train error of each model
    each_trainError[i]=np.mean((boston_y_train-lr.predict(boston_x_train[:,i:i+1]))**2)    
    # calculate test error of each model
    each_testError[i]=np.mean((boston_y_test-lr.predict(boston_x_test[:,i:i+1]))**2)
    if print_able:
        print('The score of',i+1,'th variable',boston.feature_names[i],'is',each_score[i])
        print('The train error of',boston.feature_names[i],'model is',each_trainError[i])
        print('The test error of',boston.feature_names[i],'model is',each_testError[i],'\n')
if draw_able:
    x_value=range(len(data[0]))
    # draw the score pic of each variable
    plt.bar(x_value, each_score, alpha = 1, color = '#87cefa', align="center")
    plt.autoscale(tight=True)
    plt.xticks([i for i in range(len(data[0]))],boston.feature_names,rotation="60")
    plt.xlabel("Feature names")
    plt.ylabel("Linear correlation score")
    plt.title("The linear correlation score of each variable")
    plt.show()
    
    # draw the train error of each variable
    plt.bar(x_value, each_trainError, alpha = 1, color = '#fa8072', align="center")
    plt.autoscale(tight=True)
    plt.xticks([i for i in range(len(data[0]))],boston.feature_names,rotation="60")
    plt.xlabel("Feature names")
    plt.ylabel("Train error of each variable")
    plt.title("The mean train error of each variable")
    plt.show()

    # draw the train error of each variable
    plt.bar(x_value, each_testError, alpha = 1, color = '#43cd80', align="center")
    plt.autoscale(tight=True)
    plt.xticks([i for i in range(len(data[0]))],boston.feature_names,rotation="60")
    plt.xlabel("Feature names")
    plt.ylabel("Test error of each variable")
    plt.title("The mean test error of each variable")
    plt.show()
    


# In[7]:


# predict target with all variable
lr.fit(boston_x_train,boston_y_train)
# calculate union linear correlation score
union_score=lr.score(boston_x_test,boston_y_test)
# calculate train error of union model
union_trainError=np.mean((boston_y_train-lr.predict(boston_x_train))**2) 
# calculate test error of union model
union_testError=np.mean((boston_y_test-lr.predict(boston_x_test))**2)
if print_able:
    print('The union score is',union_score)
    print('The union train error is',union_trainError)
    print('The union test error is',union_testError)
    

