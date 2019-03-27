# Reinforcemant Learning

Reinforcement Learning is a kind of machine learning algorithm which neither uses the supervised learning strategy nor an unsupervised learning one.

In this learning algo the **agent** learns to behave in an ideal manner by collecting the reward from the **environment**.

For an agent the entire environment is its universe and it collects the reward from the environment to change its state.

The altimate aim of agent is to collect maximum reward and become better at every step.

The environmet can provide the reward at every action taken by the agent or at the end of episode.


### Markovs decision process model

The markdown decision model contains:

* **State (S)** : Set of all possible states a agent can have


* **Model** : Gives the action effect to an agent


* **Action (A)** : Set of all actions an agent can take in a given state


* **Reward (R)** : The reward for being in state S or the reward the agent has at the end.


* **Policy (!(http://latex.codecogs.com/gif.latex?%5Cinline%20%24%5Cpi%24))** : The solution to markovs decision process or the behaviour of an agent in a marticular state. It is the mapping between all observation/state of an agent to the action to be taken by agent.

#### MDP notation

At each time step $t=0,1,2...$ the agent receives some representation of environment state $S_t\epsilon S$. Based on this state, the agent selects an action $A_t\epsilon A$.This gives us **state-action** pair $\left(S_t,A_t\right)$.

Time is then incremented from $t$ to $t+1$, and the environment id transitioned to a new state $S+1\epsilon S$. At this time the agent receives a **numeric Reward** $R_{t+1}\epsilon R$ for the action $A_t$ taken from state $S_t$.

We can think of the process of receiving a reward as an arbitrary function $f$ that maps state-action pairs to rewards. At each time $t$, we gave

\[
f\left(S_t,A_t\right) = R_{t+1}
\]

Let's analyze it step by steps:

1. At time $t$, the environment is in state $S_t$.
2. The agent obsetves the current state and selects action $A_t$.
3. The environment transitions to state $S_{t+1}$ and grants the agent reward $R_{t+1}$.
4. This process then starts over for the next time step, $t+1$.
    * **Note**, $t+1$ is no longer in the future, but now is the present.

### Transition Probabilities

The reward we get has always a relation with the state and action pair so there is always a probability distribution over $S$ and $R$. Also these distribution depend on the **previous** state aciton that have occurred.

For example, suppose $s^{'}\epsilon S$ and $r\epsilon R$. Then there is **some** probability that $S_t=s^{'}$ and $R_t=r$. This probability is determined by the particular values of the **preceding** state $s\epsilon S$ and the action $a\epsilon A\left(s\right)$. Note that $A\left(s\right)$ is the set of actions that can be  taken from state $s$.

### Expected Return

The goal of an agent in MDP is to maximize **cumulative reward**.This concept is called **Expected Return** and is defined by $G$.
The expected return at a time interval $t$ can be defined as


$G_t=R_{t+1}+R_{t+2}+....+R_{T}$

Where $T$ is the final time step.

**Note:** The concept of expect return is very important as it
1. Guides the agent to achieve optimal solution
2. The agent dosen't not only look for immediate reward but also focuses to atain maximum reward, which could only be achieved by optimal solution.


### Episodic vs Continuing Task

Episode is the time interval between the inital state of the environment form where the agent starts to a termination state, this time interval above is described as $T$. Suppose there is no terminal state then the time interval $T$ becomes infinite and the cumulative reward will always tend to infinity and so there is no room for the agent to learn enything from the environment.
To overcome such problem we introduce a new term the **discount factor** $\gamma$ between 0 and 1, and redefine the above cummulative reward function as
\[
G_t=\gamma R_{t+1}+\gamma^{2}R_{t+2}+....+\gamma^{T}R_{T}\\
= \sum_{k=0}^{\infty} \gamma^kR_{t+k+1}
\]

**This new defination of cumulative reward will make our agent to care more in nearby cummulative reward then commulative future reward**

### Policies and value function

**Policy$\left(\pi\right)$** is the behaviour of agent when it is in state $S$, or it is the liklihood for an agent in a state $S$ to take a action $A$, the more liklihood the more is the probability of and agent to choose an action from $A$.

The probability of taking an action in a given state is :
\[
\pi\left(a|s\right)
\]


**Value Function** is the measure of how good the current state is for the agent. In terms of reward selecting an action over other action may increase or decrease the cummulative reward for the agent. Knowing this cummulative reward in advance make our agent more powerful in choosing an action.
This term how good is represented in **expected reward**.
Since an agent takes an action which leads to a new state the **Value Function** is  always defined with respect to a **policy**.

#### State value function

\[
\begin{aligned}
 \vartheta_\pi\left(S \right){}={}& E_\pi [G_t | S_t = s]\\
                                   & = E_\pi\biggl[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}|S_t=s\biggr]
\end{aligned}
\]

_How probable is it for an agent to choose an action?_ ->            **Policy**
_How good is any given action or any state for an agent?_ -> **Value Function**


#### Action-value function or Q-function

Similarly, we define the action-value function in the same way as

\[
\begin{aligned}
 q_\pi\left(s,a \right){}={}& E_\pi [G_t | A_t = a]\\
                                   & = E_\pi\biggl[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}|S_t=s,A_t=a\biggr]
\end{aligned}
\]
**This is called Q-function and the output of this is called Q-Value**

#### Optimal Policies (How does model learns)

The goal of a reinforcement algorithm is to find a policy that makes maximum reward from its environment. A policy is discarded and replaced with a new policy if the new policy yields a better return than the previous.
So in terms of policy we can say,
\[
\pi \ge \pi^{'} if\,and\,only\,if\,\vartheta_\pi \left(S\right) \ge \vartheta_{\pi^{'}}\left(s\right)\,for\,s\,\in S.
\]
A policy which is better than or at least same as other policies is **optimal policy**.

#### Optimal state-value Function
Defined as :
\[
\vartheta_* = max\,\vartheta_\pi\left(s\right) \:for \,all\,s\in S.
\]
The $\vartheta_*$ gives the **largest expected return** by any policy $\pi$ for each state.
#### Optimal action-value Function   $q_*$
Defined as :
\[
q_*\left(s,a\right) = max\,q_\pi\left(s,a\right) \:for \,all\,s\in S, \:for \,all\,a\in A\left(S\right).
\]
The $q_*$ gives the **largest expected return** by any policy $\pi$ for each possible **action pair**.


### Q-Learning

Q-learning is a model-free reinforcement learning algorithm. The goal of Q-learning is to learn a policy, which tells an agent what action to take under what circumstances.

In this form of learnig the agent uses **Q-Values** to iteratively improve its behavior.


1. **Q-values or action values Q(S,A)** : Defined for states these are the actions to be taken by agent when it is in state S.


2.  **Rewards and episode** : The rewards are same as the markovs decision process.The agent starts from a state, keeps collecting rewards form the environment an ultimately reaches a state where no aciton can be state. The time series form the start to the end state is an episode, the end stae can be due to to agent falling to a situation where it can't change any other state or it  has sccessfully completed the environment requirements.


3. **Q-Table** : This table contains all the observation/states in an environment and all its crossponding action that can be taken. Out of all the action in the Q-value table at state S the one which is chosen is the action with maximum value.


4. **Learning from the environment or the greddy policy approch or the bellman approch** : The ultimate aim of the agent is to find the most suitable behavior which leads to the ideal solution. The agent does this by performing the action, getting the reward form the environment and then updating the crossponding action placeholder in Q-value table.

      **Current value = Immediate value + Future rewards**
\[
Q\left(s_t,a_t\right) = \left(1-\alpha\right) + \alpha \cdot\left(r_t + \gamma \cdot maxQ_a\left(s_{t+1},a\right)\right)
\]

    $\alpha$ is the learning rate <br />


    $\gamma$ is the discount factor. This gives weightage to the future outcome<br />


    $r_t$ is the reward


    $ maxQ_a\left(s_{t+1},a\right)$ This is the future expected reward

#### Randomness Vs Prediction
During initial phases we give more emphasis on the randomness so that our agent gets more time to explore the surrounding and then we gradually decrease it so that our agent gradually gravitates towards the ideal solution.
