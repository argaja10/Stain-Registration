# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 13:04:36 2024

@author: bhargavghanekar
"""
import os

def generate_tree(directory, prefix=''):
    # Contents of the current directory
    contents = os.listdir(directory)
    # Sort the contents so that directories come first
    contents.sort()
    contents = sorted(contents, key=lambda x: os.path.isfile(os.path.join(directory, x)))

    for i, name in enumerate(contents):
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            print(f"{prefix}├── {name}")
            generate_tree(path, prefix + "│   ")
        else:
            print(f"{prefix}└── {name}")



generate_tree('C:/Users/bhargavghanekar/Downloads/stain_registration_v2-main/stain_registration_v2-main')