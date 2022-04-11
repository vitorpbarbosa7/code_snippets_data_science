import os 
from contextlib import contextmanager

cwd = os.getcwd()
os.chdir('dirone')
print(os.listdir())
os.chdir(cwd)

cwd = os.getcwd()
os.chdir('dirtwo')
print(os.listdir())
os.chdir(cwd)

@contextmanager
def change_dir(destination:str):
    try:
        cwd = os.getcwd()
        os.chdir(destination)
        yield 
    finally:
        os.chdir(cwd)

with change_dir('dirone'):
    print(os.listdir())

with change_dir('dirtwo'):
    print(os.listdir())
