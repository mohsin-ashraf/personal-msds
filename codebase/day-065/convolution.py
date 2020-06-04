import numpy as np

input_image = np.random.randn(250,250)
kernal = np.random.randn(3,3)

output_height = input_image.shape[0] - kernal.shape[0] + 1
output_width = input_image.shape[1] - kernal.shape[1] + 1
output_image = np.zeros((output_height,output_width))

for i in range(0,output_height):
    for j in range(0,output_width):
        for ii in range(0,kernal.shape[0]):
            for jj in range(0,kernal.shape[1]):
                output_image[i,j] += input_image[i+ii,j+jj] * kernal[ii,jj]


print ('Convolution Done')
