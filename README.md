# ChatTime
ChatTime is a real-time messaging application that aims to use machine learning to detect cyberbullying and offensive language, protecting users from a harmful online environment by filtering out negative messages. ChatTime uses machine learning models to apply sentiment analysis and detect facial expressions, allowing the app to determine the emotion and intent of a given message.

This project is an APS360 course project. Our implementation of our machine learning models can be found in the Jupyter notebooks on the master branch.

## Getting Started
ChatTime uses GRU and AlexNet transfer learning, taking in text input via keyboard and facial data via webcam to determine if a sender is sending a message with the intent to harm or bully the recipient. Negative messages sent with negative emotions will be blocked, and a pop-up warning will be sent to the recipient. Warning pop-ups will be sent if the message is negative, but the facial expression is neutral and past messages are also negative.

Our app allows clients to connect with a server locally or over a common network. By specifying a host and port, a connection can be established. A GUI will pop-up for clients to begin chatting.

### Installing and Setting Up Folders
Our app runs with Python 3.7 on Anaconda. Users will need to install OpenCV, NumPy, PyTorch, Pillow, and matplotlib.

Users will need to set up a folder that will be shared by the server and client. It is recommended that the server is launched from this directory. In this folder, users will also need to save the ```test_0843_model_alexnet_ann_bs128_lr0_001_epoch149``` file, taken from GitHub, and a DropBox file, accessible by [clicking here](https://www.dropbox.com/s/395hb1x5xnmc4dg/model_RNN_GRU_bs100_lr0.003_epoch6?dl=0).

Photos taken via webcam will also be saved (by the client) to this folder, and accessed and processed by the server. However, please note, photos will be deleted immediately after use to ensure privacy.

### Running
The following terminal command starts up the server:
```
python App_Server.py

Enter host: 127.0.0.1
Enter port: 33000
```

This command, on a separate terminal, or separate machine (depending on what host IP is chosen), will start up a client. Enter the host and port to connect:
```
python App_Client.py

Enter host: 127.0.0.1
Enter port: 33000
```
A GUI will pop up to allow the user to begin chatting.
