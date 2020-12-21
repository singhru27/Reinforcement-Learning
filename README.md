Project Overview:

This project uses Reinforcement Learning to solve the taxi and mountain car Open AI gym problems. For taxi, the objective can be explored in detail at (https://gym.openai.com/envs/Taxi-v2/). For mountain car, the objective can be explored in detail at (https://gym.openai.com/envs/MountainCar-v0/).

To solve the taxi problem, I have implemented the SARSA and SARSA Lambda algorithms. For the mountain car problem, I have incorporated SARSA Lambda with function approximation (using the Fourier basis functions).

Instructions to run:

```bash
python3 taxi_sarsa.py
```

will execute the SARSA learning algorithm and save the q-values and policy to .npy files. . Note: There is already a default set of  values/policy included in the repository from a previous learning sequence. Running the above command will overwrite these files. This will also create a PNG file with the learning curve for this environment. 

```bash
python3 taxi_sarsa.py test
```

will use the saved policies over ten iterations of the taxi environment and print out the rewards (steps to completion).

```bash
python3 taxi_sarsa_lambda.py
```

will execute the SARSA Lambda learning algorithm and save the q-values and policy to .npy files. Note: There is already a default set of values/policy included in the repository from a previous learning sequence. Running the above command will overwrite these files. This will also create a PNG file with the learning curve for this environment. 

```bash
python3 taxi_sarsa.py test
```

will use the saved policies over ten iterations of the taxi environment and print out the rewards (steps to completion)

```bash
python3 mountain_car_taxi_fourier.py
```

will execute the SARSA Lambda learning algorithm with function approximation for the mountain car environment and save the learned weights .npy files. Note: There is already a default set of weights included in the repository from a previous learning sequence. Running the above command will overwrite these files. This will also create a PNG file with the learning curve for this environment. 

```bash
python3 taxi_sarsa.py test
```

will use the saved policies over ten iterations of the mountain car environment and print out the rewards (steps to completion)