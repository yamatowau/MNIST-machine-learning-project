#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import torch
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
model = Net()
model.load_state_dict(torch.load(r"C:/Users/shery/syminiproject/model.pth", map_location=torch.device('cpu')))
model.eval()

st.title("MNIST Digit Recognizer")
st.markdown('''
Try to write a digit!
''')

SIZE = 192

canvas_result = st_canvas(
    fill_color="#ffffff",
    stroke_width=10,
    stroke_color='#ffffff',
    background_color="#000000",
    height=150,width=150,
    drawing_mode='freedraw',
    key="canvas",
)

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    img_rescaling = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Input Image')
    st.image(img_rescaling)

if st.button('Predict'):
    #convert image from colour to gray
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #change the values from [0,255] to [0,1] then convert the numpy array to a torch tensor
    bwx = torch.from_numpy(test_x.reshape(1, 28, 28)/255)
    #change input to float from double then unsqueeze changes from (1,28,28) to (1,1,28,28)
    val = model(bwx.float().unsqueeze(0))
    #result will be a one-hot tensor
    st.write(f'result: {np.argmax(val.detach().numpy()[0])}')
    #display the one-hot tensor output
    st.bar_chart(np.exp(val.detach().numpy()[0]))
    print(np.exp(val.detach().numpy()[0]))


# In[ ]:




