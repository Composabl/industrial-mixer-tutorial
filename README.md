The Industrial Mixer Use Case
The industrial mixer use case is a realistic case study of an agent controlling a continuous stirred tank chemical reaction. Read more about the case study and the agents that solved it in our whitepaper Use Cases for Intelligent Agents: Industrial Mixer.

Use Case Summary
The industrial mixer agent controls the temperature in a tank where a chemical reaction occurs to create a product.

![image](https://github.com/user-attachments/assets/c93485fb-71e5-4be9-98bb-9ac53dfa338a)


As the chemicals are stirred together in the tank, the reaction produces heat at a nonlinear, unpredictable rate. If the tank isn’t cooled enough, it can reach dangerous temperatures, a condition called thermal runaway. If it’s cooled too much, less product will be produced. The agent needs to balance these two goals, keeping the tank at the right temperature at every moment to maximize production while ensuring safety.

Two Competing Goals
As in all Machine Teaching use cases, the "fuzziness" or nuance in this process can be summarized in the form of two separate goals that must be balanced against each other:

Produce as much product as possible
Eliminate the risk of thermal runaway
The key to balancing these goals is maintaining the right temperature in the tank throughout the reaction, so that it's hot enough to be efficient but cool enough that the thermal runaway threshold is never crossed.

Controlling the Temperature in the Tank
This use case has only one control variable. The agent controls the termperature in the tank by adjusting the temperature of the mixture using a jacket filled with coolant.

![image](https://github.com/user-attachments/assets/9b65fb7e-e06b-4e2b-959b-00afb54c5fbf)


If the chemicals get too hot and approach thermal runaway, the coolant temperature can be decreased to bring down the temperature in the tank – but the conversion rate will also decrease.

Three Different Phases with Different Control Needs
One of the reasons this use case is complex is that it occurs in three different phases.

It starts in a steady state with low temperature and low productivity
It goes through a transition period when the temperature can change quickly and unpredictably
It ends in a steady state of high but consistent temperature and high productivity
The transition phase is the most unpredictable and challenging to control, with the highest risk of thermal runaway.
