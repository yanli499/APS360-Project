#!/usr/bin/env python3
"""Server for multithreaded (asynchronous) chat application."""
"""From https://medium.com/swlh/lets-write-a-chat-app-in-python-f6783a9ac170"""
import os

from socket import AF_INET, socket, SOCK_STREAM
from threading import Thread
from ML_API_V1 import ModelsContainer
from pathlib import Path

def accept_incoming_connections():
    """Sets up handling for incoming clients."""
    while True:
        client, client_address = SERVER.accept()
        print("%s:%s has connected." % client_address)
        client.send(bytes("Hi! Please type your name and press enter to begin!", "utf8"))
        addresses[client] = client_address
        Thread(target=handle_client, args=(client,)).start()


def handle_client(client):
    # Takes client socket as argument.
    """Handles a single client connection."""
    name = client.recv(BUFSIZ).decode("utf8")
    welcome = 'Welcome %s! If you want to quit the app, type {quit} to exit.' % name
    client.send(bytes(welcome, "utf8"))
    msg = "%s has joined the chat!" % name
    broadcast(bytes(msg, "utf8"))
    clients[client] = name

    while True:
        msg = client.recv(BUFSIZ)
        if msg != bytes("{quit}", "utf8"):
            broadcast(msg, name+": ")
        else:
            client.send(bytes("{quit}", "utf8"))
            client.close()
            del clients[client]
            broadcast(bytes("%s has left the chat." % name, "utf8"))
            break


def broadcast(msg, prefix=""):
    # prefix is for name identification.
    """Broadcasts a message to all the clients."""

    # RUN ML MODEL HERE -----------
    # <Get screenshot + save to some folder, returning filepath>
        # image_filepath = <screenshot related code>
    img_path = os.path.normpath('/Users/Harshita/Documents/GitHub/ChatTime/frame.jpg')
    bully = ML_MODELS.combine_results(img_path)

    for sock in clients:
        sock.send(bytes(prefix, "utf8")+msg)


# SERVER GLOBAL VARIABLES
clients = {}
addresses = {}

HOST = '127.0.0.1'
PORT = 33000
BUFSIZ = 1024
ADDR = (HOST, PORT)

SERVER = socket(AF_INET, SOCK_STREAM)
SERVER.bind(ADDR)

ML_MODELS = ModelsContainer()

if __name__ == "__main__":
    SERVER.listen(5)
    print("Waiting for connection...")
    ACCEPT_THREAD = Thread(target=accept_incoming_connections)
    ACCEPT_THREAD.start()
    ACCEPT_THREAD.join()
    SERVER.close()
