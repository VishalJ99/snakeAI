# Snake Ai - using deep Q learning

This project is about using a branch of reinforcement learning (deep Q learning) to teach an AI how to play the game snake. 

NOTE: The code for the game.py and model.py files was modified from the repo https://github.com/patrickloeber/snake-ai-pytorch/.

Here is a blog post explaining the underlying algorithm (unfinished...)



## Set up
1) 		git clone https://github.com/VishalJ99/snakeAI.git	
2)		cd snakeAI
3) 		python -m venv snakeAI    	# mac virtualenv snakeAI
4)		snakeAI\Scripts\activate.bat # mac source snakeAI/bin/activate 
5)	 	pip3 install -r requirements.txt
6) 		python agent.py

It will take about 150 episodes / 10 minutes of training before the agent starts reliably getting scores > 10.
