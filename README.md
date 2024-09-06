# captchas-recognition

## Background
A website uses Captchas on a form to keep the web-bots away. However, the captchas it generates, are quite similar each time:
- the number of characters remains the same each time  
- the font and spacing is the same each time  
- the background and foreground colors and texture, remain largely the same
- there is no skew in the structure of the characters.  
- the captcha generator, creates strictly 5-character captchas, and each of the characters is either an upper-case character (A-Z) or a numeral (0-9).

## Input and Output
The captcha images and the Captchas' text are in the input and output folder of data folder. Each captch image contains 5 character.

## Goal
The goal of this task is to build a AI model to identify the unseen captchas. 

## Process
The process of building a AI model to infer the unseen captchas is as follows:

1. Data processing
- Read the gray image and used a binary threshold to invert the image
- Find contours (these will be the outlines of the characters), and loop over the contours and extract individual characters, transform the 2D arrays of individual characters into 1D verctors, 
  use them as input features for model training
- Read the corresponding text of each captcha images, and use them as target for model training

2. Model training
- Build a pipeline, to (1) standardize the data; (2) PCA to reduce the dimensions of the data; (3) Kmeans to cluster data into 36 clusters (26 characters and 10 numerals)
- The trained pipeline is saved in the model folder as well as the mapping of cluster labels to characters and numerals

3. Inference
- By loading the saved pipeline, we can predict the cluster labels of the 5-character captchas in the captcha image, and use the saved mapping relations to get the final output. 
