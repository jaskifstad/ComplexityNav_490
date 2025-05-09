# Training and Tuning
## MPC and MPPI
The MPC and MPPI method cost function parameters are tuned via a grid search over the sigma h, s, and r parameters in the range [0.3, 2.1] in intervals of 0.3. The q parameters were fixed manually to (100.0, 1.0, 5.0, 1.0) and kept constant across methods. The additional MPPI parameters (noise covariance diagonal, samples, and horizon) were also tuned using a separate grid search with values (1.0, 2.0, 3.0, 4.0, 5.0), (100, 250, 500, 1000), (3, 4, 5, 6, 7), rejecting combinations that led to a control loop operating slower than 0.25Hz (the simulation timestep). All parameters were tuned in a passing and crossing scenario with 5 ORCA and 5 SFM agents in a 10x10m workspace using 100 trials to calculate metrics.

## RGL
RGL training was attempted in scenarios with (5, 10, 15) (ORCA, SFM, ORCA and SFM) agents in a 12x12m workspace with circle crossing directionality. We found the training did not converge to a generalized collision avoiding policy in all cases except for the 5 agent ORCA scenario, so that is the model used in experiments.

# Simulation Framework
## Environment
The environment contains n+1 agents. N of them are humans controlled by certain unknown
policy. The other is robot and it's controlled by one known policy.
The environment is built on top of OpenAI gym library, and has implemented two abstract methods.
* reset(): the environment will reset positions for all the agents and return observation 
for robot. Observation for one agent is the observable states of all other agents.
* step(action): taking action of the robot as input, the environment computes observation
for each agent and call agent.act(observation) to get actions of agents. Then environment detects
whether there is a collision between agents. If not, the states of agents will be updated. Then 
observation, reward, done will be returned.


## Agent
Agent is a base class, and has two derived class of human and robot. Agent class holds
all the physical properties of an agent, including position, velocity, orientation, policy and etc.
* visibility: humans are always visible, but robot can be set to be visible or invisible
* sensor: can be either visual input or coordinate input
* kinematics: can be either holonomic (move in any direction) or unicycle (has rotation constraints)
* act(observation): transform observation to state and pass it to policy


## Policy
Policy takes state as input and output an action. Current available policies:
* ORCA: compute collision-free velocity under the reciprocal assumption
* CADRL: learn a value network to predict the value of a state and during inference it predicts action for the most important human
* LSTM-RL: use lstm to encode the human states into one fixed-length vector
* SARL: use pairwise interaction module to model human-robot interaction and use self-attention to aggregate humans' information
* OM-SARL: extend SARL by encoding intra-human interaction with a local map


## State
There are multiple definition of states in different cases. The state of an agent representing all
the knowledge of environments is defined as JointState, and it's different from the state of the whole environment.
* ObservableState: position, velocity, radius of one agent
* FullState: position, velocity, radius, goal position, preferred velocity, rotation
* DualState: concatenation of one agent's full state and one another agent's observable state
* JoinState: concatenation of one agent's full state and all other agents' observable states 


## Action
There are two types of actions depending on what kinematics constraint the agent has.
* ActionXY: (vx, vy) if kinematics == 'holonomic'
* ActionRot: (velocity, rotation angle) if kinematics == 'unicycle'
