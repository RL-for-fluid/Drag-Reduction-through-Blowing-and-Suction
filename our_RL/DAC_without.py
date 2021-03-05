from printind.printind_function import printi, printiv
from env import resume_env

import numpy as np
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
import env
import os
import csv

printi("resume env")

environment = resume_env(dump=100000, single_run=False)
deterministic=True

printiv(environment.states)
printiv(environment.actions)

#======================= original below =========================#

# module
import numpy as np
import pickle
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression
import time

#=== test ===#
#with open('drag.pickle','rb') as f:
#    drag=np.array(pickle.load(f))
#
#with open('lift.pickle','rb') as f:
#    lift=np.array(pickle.load(f))
#    
#with open('act.pickle','rb') as f:
#    act=np.array(pickle.load(f))
#    
#for i in range(drag.shape[0]//2):
#    print(float(act[i+drag.shape[0]//2]))#, float(lift[i]), float(act[i]))#[i,0]

#with open('J.pickle','rb') as f:
#    J=np.array(pickle.load(f))
#
#for i in range(J.shape[0]):
#    print(float(J[i]))


max_batch = 1000       #episode
n_step = 200             # n_step TD error, dimensionless time = 1
dim_obs = 11           # dimension of observe
dim_act = 1           # dimension of action
dim_the = dim_obs    # dimension of theta
gam = 0.9              # Discount rate

#== Fatal parameter ==#
batch_size = 4000            # DNS time t=20
alpha_t_base=1e-3            # policy parameter
alpha_t = alpha_t_base       # policy parameter
num_train_episode = 10       # the number of episodes for fitting Q


gamma_list = [1.]
num_components=100
scaler_observe = sklearn.preprocessing.StandardScaler()
feature = sklearn.pipeline.FeatureUnion([
    (str(i), RBFSampler(gamma=gamma_list[i],
                       n_components=num_components))
    for i in range(len(gamma_list))
])


def minimize_TDerror5(obs_np,the_np,act_np,rew_np,phi_imp_np,n_step,gam,rcond=1e-15):
#====  divied===#
    obs0,the0,act0,phi0 = obs_np[:-n_step],the_np[:-n_step],act_np[:-n_step],phi_imp_np[:-n_step]
    obs1,the1,act1,phi1 = obs_np[n_step:], the_np[n_step:], act_np[n_step:], phi_imp_np[n_step:]


#=== preparation ===#
    p=gam*phi1-phi0
    
    rew0=np.zeros(len(rew_np)-n_step)
    gams=gam**(np.arange(n_step)/n_step)
    for n in range(len(rew0)):
        rew0[n]=np.sum(rew_np[n:n+n_step]*gams) 
    
#=== setting ===# 
    A = np.matmul(p.T,p)
    Y =-np.matmul(rew0,p)

#=== solve ===#
    #X=np.linalg.solve(A,Y)
    A2=np.linalg.pinv(A,rcond=rcond)
    X=np.matmul(A2,Y)

#=== arrangment ===#
    TDerr=(((np.matmul(p,X)+rew0)**2).mean()/(np.matmul(phi0,X)**2).mean())**0.5
    #print(TDerr)
    
#=== Value ===#    
    Value=np.matmul(phi0,X)

#=== gradient ===#
    #lr = LinearRegression(fit_intercept=True)
    #lr.fit(the0, Value)                         
    #dthe = lr.coef_*(1-gam**(1./n_step))   
    
    #== expand ==#
    one=np.ones(the0.shape[0]).reshape(-1,1)
    the0_expand=np.concatenate([the0, one], 1)

    #== setting ==#
    A = np.matmul(the0_expand.T,the0_expand)
    Y = np.matmul(Value,the0_expand)

    #=== solve ===#
    A2=np.linalg.pinv(A,rcond=rcond)
    X=np.matmul(A2,Y)
    dthe=X[:the0.shape[1]]*(1-gam**(1./n_step)) 
    
    return dthe,TDerr #w_weight,v_weight#,A,Y,X,A2

    
    
#=== Observe Action sequence setting ===#
act = np.zeros(dim_act)
action = np.zeros(dim_act*2)
obs = np.zeros(dim_obs)

#state = environment.reset()
#environment.render = True
#obs = state


#=== Policy parameter initialization ===#
np.random.seed(seed=0)
the=np.zeros(dim_the) #np.random.normal(loc=0,scale=1e-3,size=dim_the)
dthe=np.random.normal(loc=0,scale=1e-3,size=dim_the) #np.zeros(dim_the)


# == history == #
J_total_history = []         # Average reward in mini-batch
the_train_history = []       
obs_train_history = []
act_train_history = []
rew_train_history = []
drag_train_history = []
lift_train_history = []

TDerr_history = []
alpha_history = []
dthe_history = []

TDerr,TDerr2,TDerr3 = 0,0,0
DNS_time ,Of_time,data_time,np_time,feature_time=0,0,0,0,0
Qfit1_time,Qfit2_time,Qfit3_time,gradient_time,adjust_time,update_time,plot_time,pickle_time=0,0,0,0,0,0,0,0


#=== learning starts ===#
for i_episode in range(1, max_batch + 1):  
    total1 = time.time()##########################################################
    t1=time.time()##########################################################

   
    #========== Perform DNS ==========#
    t1=time.time()##########################################################
    state = environment.reset()
    environment.render = True
    obs = state
    for steps in range(batch_size):
        #=== for history ===#
        obs_train_history.append(np.array(obs)) # Save total history of observe_t
        
        #=== Update policy parameter ===#
        #if i_episode <= num_train_episode+1 and steps==0 and i_episode>1:
        #    dthe=np.random.normal(loc=0,scale=1e-3,size=dim_the)        
        #the += dthe*alpha_t
        
        #=== Determine action by policy ===#
        if steps % 50 == 0:
            the += dthe*alpha_t
            act[0] = np.dot(obs,the)
        action[0] = 0.01*np.tanh( act[0])
        action[1] = 0.01*np.tanh(-act[0])
        state, _, rew = environment.execute(action) #, drag, lift
        obs=state
        #obs[0] = state[0]
        #obs[1] = state[1]
        #obs[2] = state[2]
        #obs[3] = state[3]
        #obs[4] = state[4]
        #k=5
        #for i in range(5):
        #    for j in range(i+1):
        #        obs[k] = state[i]*state[j]
        #        k+=1
        #print(k)
        #obs[5] = state[0]*state[0]
        #obs[6] = state[1]*state[1]
        #obs[7] = state[2]*state[2]
        #obs[8] = state[3]*state[3]
        #obs[9] = state[4]*state[4]
         
         
                
        #=== for history ===#
        the_train_history.append(np.array(the)) # Save total history of theta
        act_train_history.append(np.array(act)) # Save total history of action_t
        rew_train_history.append(np.array(rew)) # save history of reward_{t+1}
        #drag_train_history.append(np.array(drag))
        #lift_train_history.append(np.array(lift))
    t2=time.time()##########################################################
    DNS_time=t2-t1##########################################################

    
#=== Object function ===#
    t1=time.time()##########################################################
    J=np.mean(np.array(rew_train_history)[-batch_size:])
    J_total_history.append(J)
    t2=time.time()##########################################################
    Of_time=t2-t1##########################################################
    
#==== training data ====#    
    t1=time.time()##########################################################
    if len(obs_train_history)==batch_size*(num_train_episode+1):
        del obs_train_history[:batch_size]
        del rew_train_history[:batch_size]
        del the_train_history[:batch_size]
        del act_train_history[:batch_size]
    #print("check obs", np.array(obs_train_history).shape)
    #print("check rew", np.array(rew_train_history).shape)
    #print("check the", np.array(the_train_history).shape)
    #print("check act", np.array(act_train_history).shape)
    t2=time.time()##########################################################
    data_time=t2-t1##########################################################
    
#========= Reinforcement learning started =========#    
    #if i_episode >= num_train_episode+1:
    t1=time.time()##########################################################
    obs_np=np.array(obs_train_history)
    rew_np=np.array(rew_train_history)
    the_np=np.array(the_train_history)
    act_np=np.array(act_train_history)
    t2=time.time()##########################################################
    np_time=t2-t1##########################################################

#========= Acquisition of features =========# 
    t1=time.time()##########################################################
    obs_the_np=np.concatenate([obs_np, the_np], 1)
    #obs_the_scale_np = (obs_the_np - obs_the_np.mean(0)) / obs_the_np.std(0)
    phi_imp_np = feature.fit_transform(obs_the_np)
    t2=time.time()##########################################################
    feature_time=t2-t1##########################################################
    
#==== Q fitting and policy parameter ====#      
    t1=time.time()##########################################################
    dthe,TDerr=minimize_TDerror5(obs_np,the_np,act_np,rew_np,phi_imp_np,n_step,gam)
    #alpha_t=1e-4/TDerr*alpha_t_base
                
    t2=time.time()##########################################################
    Qfit3_time=t2-t1##########################################################

    #=== for history ===#
    dthe_history.append(dthe)
    TDerr_history.append(TDerr)
    alpha_history.append(alpha_t)
          
#=== Saving experience and objective function ===#
    t1=time.time()##########################################################
    #with open('obs.pickle', 'wb') as f:
    #    pickle.dump(obs_train_history,f)
    #with open('act.pickle', 'wb') as f:
    #    pickle.dump(act_train_history,f)
    with open('J.pickle', 'wb') as f:
        pickle.dump(J_total_history,f)
        
    with open('TDerr.pickle', 'wb') as f:
        pickle.dump(TDerr_history,f)
    with open('alpha.pickle', 'wb') as f:
        pickle.dump(alpha_history,f)
    with open('dthe.pickle', 'wb') as f:
        pickle.dump(dthe_history,f)

    #with open('drag.pickle', 'wb') as f:
    #    pickle.dump(drag_train_history,f)
    #with open('lift.pickle', 'wb') as f:
    #    pickle.dump(lift_train_history,f)
    #with open('the.pickle', 'wb') as f:
    #    pickle.dump(the_train_history,f)
    #with open('rew.pickle', 'wb') as f:
    #    pickle.dump(rew_train_history,f)
    t2=time.time()##########################################################
    pickle_time=t2-t1##########################################################
    
#=== monitoring ===#        
    print("#====================",i_episode, " episode ====================#")
    print("J       = ", J           )
    print("dtheta  = ", dthe        )
    print("theta   = ", the         )
    print("TDerr   = ", TDerr       )
    print("alpha_t = ",alpha_t      )
    print("fitting Ep.", np.array(rew_train_history).shape[0], "/", batch_size*num_train_episode)
    #print("w       = ", w_weight)
    #print("v       = ", v_weight)
    
#=== Time related ===#   
    print("#==================== TIME ====================#")
    print("DNS              : ", DNS_time       )
    print("Object function  : ", Of_time        )
    print("Train data       : ", data_time      )
    print("Create numpy     : ", np_time        )
    print("Feature          : ", feature_time   )
    #print("Q fit1           : ", Qfit1_time     )
    #print("Q fit2           : ", Qfit2_time     )
    print("Q fit3           : ", Qfit3_time     )
    #print("Policy gradient  : ", gradient_time  )
    #print("Adjust dtheta    : ", adjust_time    )
    #print("Upadate theta    : ", update_time    )
    #print("Plot             : ", plot_time      )
    print("Pickle           : ", pickle_time    )
    total2 = time.time()##########################################################
    print("#=============================================#")
    print("TOTAL TIME per minibatch : ", total2-total1  )



    


