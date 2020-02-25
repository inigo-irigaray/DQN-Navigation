# DQN-Navigation

## Environment

<p align=justify>In this project an agent navigates a vectorized environment to collect yellow bananas, obtaining a reward of +1 each, and avoid blue bananas, for a -1 reward for each blue banana collected. The state space has 37 dimensions representing the agent's velocity and ray-based perception of the objects around the agent. The action space is discrete and has four dimensions:</p>

· Action <b>0</b>: move forward.

· Action <b>1</b>: move backward.

· Action <b>2</b>: turn left.

· Action <b>3</b>: turn right.

<p align=justify>The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.</p>

<p align=justify>You can download the environment to play in your own machine from the following links:</p>

 · <a href=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip>Linux</a>
 
 · <a href=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip>MacOS</a>
 
 · <a href=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip>Windows(32-bit)</a>
 
 · <a href=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip>Windows(64-bit)</a>
 
 · <a href=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip>AWS(headless)</a>
 
 Then, place the downloaded file in bin/ and decompress the file.
 
 ## Requirements
 
 Nanodegree's prerequisites: <a href=https://github.com/udacity/deep-reinforcement-learning/#dependencies>link.</a>
 
    python==3.6
    tensorflow==1.7.1
    Pillow>=4.2.1
    matplotlib
    numpy>=1.11.0
    jupyter
    pytest>=3.2.2
    docopt
    pyyaml
    protobuf==3.5.2
    grpcio==1.11.0
    torch==0.4.0
    pandas
    scipy
    ipykernel
    tensorboardX==1.4
    unityagents
    
