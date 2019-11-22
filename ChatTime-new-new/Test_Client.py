#!/usr/bin/env python3
"""Script for Tkinter GUI chat client."""
"""From https://medium.com/swlh/lets-write-a-chat-app-in-python-f6783a9ac170"""
from socket import AF_INET, socket, SOCK_STREAM
from threading import Thread
import tkinter
import tkinter.messagebox
import cv2


def receive():
    """Handles receiving of messages."""
    print("Receive")
    while True:
        try:
            msg = client_socket.recv(BUFSIZ).decode("utf8")
            if msg[0:5] == "warn ":
                msg = msg[5:]
                tkinter.messagebox.showinfo("Warning", "You may feel bullied! Please exit the chatroom if you do!")
                msg_list.insert(tkinter.END, msg)
            elif msg[0:6] == "bully ":
                msg = msg[6:]
                filtered_msg = "A harmful message sent by {} was blocked!".format(msg)
                tkinter.messagebox.showinfo("Bully Alert", filtered_msg)
            else:
                msg_list.insert(tkinter.END, msg)
        except OSError:  # Possibly client has left the chat.
            break


def send(event=None):  # event is passed by binders.

    msg = my_msg.get()
    my_msg.set("")  # Clears input field.

    cam = cv2.VideoCapture(0)
    cv2.namedWindow("ChatTime")

    #should be in the client send() function
    #takes a picture everytime the user sends a message
    ret, frame = cam.read()
    cv2.imshow("ChatTime", frame)
    if not ret:
        return

    img_name = "frame.jpg"
    cv2.imwrite(img_name, frame)

    cam.release()
    cv2.destroyAllWindows()

    client_socket.send(bytes(msg, "utf8"))

    if msg == "{quit}":
        client_socket.close()
        top.quit()
    #"""Handles sending of messages."""
    #msg = my_msg.get()
    #my_msg.set("")  # Clears input field.
    #client_socket.send(bytes(msg, "utf8"))
    #if msg == "{quit}":
      #  client_socket.close()
      #  top.quit()


def on_closing(event=None):
    """This function is to be called when the window is closed."""
    my_msg.set("{quit}")
    send()


top = tkinter.Tk()
top.title("ChatTime")

messages_frame = tkinter.Frame(top)
my_msg = tkinter.StringVar()  # For the messages to be sent.
my_msg.set("Type your messages here.")
scrollbar = tkinter.Scrollbar(messages_frame)  # To navigate through past messages.
# Following will contain the messages.
msg_list = tkinter.Listbox(messages_frame, height=20, width=75, yscrollcommand=scrollbar.set)
scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)
msg_list.pack(side=tkinter.LEFT, fill=tkinter.BOTH)
msg_list.pack()
messages_frame.pack()

entry_field = tkinter.Entry(top, textvariable=my_msg)
entry_field.bind("<Return>", send)
entry_field.pack()
send_button = tkinter.Button(top, text="Send", command=send)
send_button.pack()

top.protocol("WM_DELETE_WINDOW", on_closing)

#----Now comes the sockets part----
HOST = input('Enter host: ')
PORT = input('Enter port: ')
if not PORT:
    PORT = 33000
else:
    PORT = int(PORT)

BUFSIZ = 1024
ADDR = (HOST, PORT)

client_socket = socket(AF_INET, SOCK_STREAM)
client_socket.connect(ADDR)

receive_thread = Thread(target=receive)
receive_thread.start()
tkinter.mainloop()  # Starts GUI execution.
