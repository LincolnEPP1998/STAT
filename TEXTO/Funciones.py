import pandas as pd

def TEXTO(path):
  try:
    with pandas.open(path) as file:
      text=''
      for page in file:
        text += page.getText()
        print(text)
        except:
          pass
