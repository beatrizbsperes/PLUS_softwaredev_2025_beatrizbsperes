import os
import sys

#this a a comment

print(sys.path)

help("modules")

days_of_the_week = 7

def say_hello(recipient):
    print("Hello world! Hello {} ".format(recipient))
    return recipient