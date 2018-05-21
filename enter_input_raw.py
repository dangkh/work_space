import os
import time


def line_to_int(line):
  result = []
  line = line + ' '
  tmp = ''
  for x in range(len(line)):
    if line[x] <= '9' and line[x] >= '0':
      if line[x-1] == ' ':
        tmp += line[x]
    else:
      if len(tmp)>0 and (line[x] == ' ' or x == len(line)-2):
        result.append(int(tmp))
        tmp = ''
      else:
        tmp = ''
  return result

def line_to_number(line):
  result = []
  line = line + ' '
  tmp = ''
  for x in range(len(line)):
    if line[x] <= '9' and line[x] >= '0':
      if line[x-1] == ' ':
        tmp = line[x]
      elif len(tmp) > 0: tmp += line[x]
      else: tmp = ''
    elif line[x] == '.':
      if len(tmp) > 0:
        tmp += line[x]
      else: tmp = ''
    else:
      if len(tmp)>0 and (line[x] == ' ' or x == len(line)-2):
        result.append(tmp)
        tmp = ''
      else:
        tmp = ''
  return result

class get_input():
  def __init__(self, link = None, option = 0):
    print link
    self.link = link
    self.option = option
    if link != None :
      self.read_file()

  def read_file(self):
    self.file = open(self.link, 'r')
    self.number = []
    counter = 0
    while True:
      line = self.file.readline()
      if len(line) == 0: break
      counter += 1
      if self.option == 0:
        tmp = line_to_int(line)
      else:
        tmp = line_to_number(line)
      self.number.append(tmp)

  def get_number(self):
    return self.number
