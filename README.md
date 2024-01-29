Making some custom projects with what I have learned in Deep Reinforcement Learning Hands-On:

https://www.amazon.com/Deep-Reinforcement-Learning-Hands-Q-networks/dp/1788834240


----------------------------------------------------------------------------------------------------------------------------------------

Cliff_Walking_Cross_Entrophy is a reimplmentation of the Cross-Entrophy method for a different environment than cartpole. It was my first time playing with an RL project and I think it went well. Bit of a retread from the book but it solves the environment.

----------------------------------------------------------------------------------------------------------------------------------------

Cliff_Walking_Q_Learning solves the environment using tabular Q-learning. It's a messy environment because the episode only ends when a specific spot is reached, so to make training not take forever I had to truncate tests. But either way it seemed to perform terribly and then jump to totally solved once it finally explored each state. 

----------------------------------------------------------------------------------------------------------------------------------------
