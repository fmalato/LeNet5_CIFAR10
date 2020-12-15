# LeNet5_CIFAR10
Master Thesis project on Deep Reinforcement Learning.

### **Dependencies**
- tensorflow 2.3.1
- tensorforce 0.6.2
- numpy 1.19.4
- matplotlib

### How to use the agent
- **Step 1**: Run _tensorflow_train.py_ in order to train the agent.
- **Step 2**: After training, go in _tensorforce_train.py_ line 26 and change the weights file name with the one you got from Step 1.
- **Step 3**: Check parameters in _tensorforce_main.py_ (lines 17-23). Grid is set to be 2x2 by default.
- **Step 4**: Run _tensorforce_train.py_. The agent should start to train indefinitely on a single random image from CIFAR10.

### Acknowledgements
Thanks @SestoAle for providing the _tracker_dense.py_ script.