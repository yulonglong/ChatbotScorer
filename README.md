Automated Scoring of Chatbot Responses in Conversational Dialogue
===================================
Using machine learning algorithms to automatically score the quality of a chatbot response. Explored many models such as support vector machines (SVM), random forests (RF), convolutional neural networks (CNN), and long short-term memory (LSTM). Also explored different ways to combine multiple human annotations into the appropriate gold/true class for training.

Full paper is published in International Workshop on Spoken Dialog System Technology (IWSDS) 2018. The paper proceeding is pending.

Please contact me at sky@u.nus.edu if you have any questions.

Please note that the experiments in this repository was done in early 2017. The rate of development of CUDA drivers and deep learning libraries (Keras, Theano, TensorFlow, etc) is very rapid. You may need to spend time debugging and getting the correct setup in order to run or replicate our experiments.

**TLDR:**  
- Assuming you are on Ubuntu 16.04 with Python 2.7, open terminal  
- `./setup.sh` (Install packages) 
- `cd embedding && ./run.sh && cd ..` (Download and Prepare word embedding)
- `./run_train_rf.sh` (Train Random Forests model)
- `./run_train_cnn.sh 0 GTX1070` (Train CNN model on GPU 0, a GTX1070)

**Requirements and Environment:**  
- Ubuntu 16.04  
- Python 2.7.12  
- CUDA 8.0
- Nvidia Driver 384.111

**Python Library Required:**  
- Keras 1.1.1
- Theano 0.8.2  
- scikit-learn 0.18.1  
- scipy 0.19.0   
- nltk 3.2.2  
- pydot 1.1.0  
- h5py 2.6.0  

Python libraries above can be installed via `pip`. Note that we only tested with the versions above, newer version might not work.

**Python libraries setup:**

If you are running a UNIX based machine, you can run the shell script `./setup.sh` to install all the required python libraries, assuming `pip` is already installed. If you prefer to install manually, please refer to the list of libraries above.

**To train neural network model**
- Execute `./run_train_cnn.sh <GPU_Number> <GPU_Name>` , e.g., `$> ./run_train_cnn.sh 0 TITANX`  
- It will train the best CNN model with TickTock dataset, optimistic ground truth
- Please make sure Nvidia CUDA is installed to be able to train the model using GPU. We emphasize again that we only tested with CUDA 8.0 and Nvidia driver 384.111. It might not work with newer version.
- For more details on the training arguments, refer to the sample `run_train_cnn.sh` shell script  
- The full training scripts for all dataset and all ground truth (e.g., optimistic, pessimistic, and averaging), please refer to `train_cnn_mot.sh` and `train_lstm_att.sh`.

**To train random forests/SVM model**
- Execute `./run_train_rf.sh`
- It will train the best Random Forest Model model with TickTock dataset, optimistic ground truth
- For more details on the training arguments, refer to the sample `run_train_rf.sh` shell script  
- The full training scripts for all dataset and all ground truth (e.g., optimistic, pessimistic, and averaging), please refer to `train_rf_svm.sh`.
