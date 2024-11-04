import os

path = './model/encoder/baseline__1231_001948'
for model in os.listdir(path):
    os.rename(os.path.join(path, model), os.path.join(path, 'Encoder_'+model))