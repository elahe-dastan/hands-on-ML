# Introduction
Though python is not a fast a language but it's the top one language works like deep learning, ML and ...,so to learn
machine learning in addition to know algorithms I have to know python and how to use it too.In this repository I want to
put the code examples of the book "Hands On Machine Learning" to practise them.

# Use GPU
conda install -c anaconda keras-gpu

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

---> zero

sudo apt install hwinfo
sudo hwinfo  --gfxcard --short

--->
graphics card:                                                  
                       nVidia VGA compatible controller

Primary display adapter: #29


go to software and updates,
go to additional drivers tab

sudo apt install gpustat
---> run
gpustat
