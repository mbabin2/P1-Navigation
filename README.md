[//]: # (Image References)
[image1]: https://raw.githubusercontent.com/mbabin2/P1-Navigation/master/images/anaconda.png "Conda"
[image2]: https://raw.githubusercontent.com/mbabin2/P1-Navigation/master/images/jupyter_home.png "Home"
[image3]: https://raw.githubusercontent.com/mbabin2/P1-Navigation/master/images/set_kernal.png "Kernel"

# Udacity "DRLN" - Project 1 Submission

## 1. Environment Details

For this project, we were tasked with training an agent to navigate a 3D environment in which to collect bananas. The reward function for the environment is a simple one, the agent receives +1 for each yellow banana collcted, -1 for each blue banana collected, and 0 otherwise. Thus the goal of this task is to collect as many yellow bananas as possible. This task is also episodic, and it requires the agent to average a score of +13 over 100 consecutive episodes in order to be considered solved.

The state space for this environment contains 37 dimensions corresponding to various ray-based perception of nearby objects, and only 4 discrete actions the agent can take: 
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

## 2. Dependencies

Inorder to run this project, please follow the instructions below:

1. Install [Anaconda](https://www.anaconda.com/download/#windows).
![Conda][image1]


2. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```


3. Clone the repository.
```bash
git clone https://github.com/mbabin2/P1-Navigation
```


4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```


5. Open the Jupyter Notebook using the following command from the `p1_navigation/` folder.
```bash
jupyter notebook
```


6. Select either `Navigation_Test.ipynb` to view a pre-trained agent, or `Navigation_Train.ipynb` to train an agent for yourself.
![Home][image2]


7. Before running any code in a notebook, make sure to change the kernel to `drlnd` under the `Kernel` menu and `Change kernel` sub-menu. 

![Kernel][image3]

## 3. The Jupyter Notebooks

Any instructions for executing the code within the Jupyter Notebooks will be included inside the notebooks themselves. Any explanations for the implementation details of algorithms will be outlined in the `Report.pdf` file found at the root of this repository.
