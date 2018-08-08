## Overhand Fitness Exercise Classification (CoderBunker Machine learning Colearners) :fire:    
<img src= 'http://overhandfitness.cn/assets/img/logo.png' aligh='left' width= '80' height= '60'> 
<img src= 'http://res.cloudinary.com/hrscywv4p/image/upload/c_limit,fl_lossy,h_300,w_300,f_auto,q_auto/v1/775700/480c7f839475496a9608f42c194299b4_jvcd7e.png' aligh= 'left' width= '80' height= '60' > 

#### 
[Overhand Fitness](http://overhandfitness.cn/en/home) 
A company that offers a gamified, immersive, full-body workout, They have fun games and activities to get you moving and burning calories. They use displays and fitness trackers to measure exercise performance.    
From their website: “Using two on-wrist devices, Overhand Fitness precisely tracks and measures a person’s body movements on the X, Y, and Z axis, such as punches (counts, speed, power), jump rope, push ups, squats, burpees, jumping jacks, and many more.  This is done using sensors and advanced algorithms able to recognize the precise inertial signature of each body movement.”      

#### Coderbunker 
[Coderbunker](http://www.coderbunker.com) is a community for developers to further develop their software development skills and create opportunities for collaboration with our customers.  

#### Prerequisites  
Building environment: 
* Python [Pandas](https://pandas.pydata.org), [Numpy](http://www.numpy.org), [Plotly](https://plot.ly/ipython-notebooks/cufflinks/), [Visdom](https://github.com/facebookresearch/visdom), [Pytorch](https://pytorch.org), [Keras](https://keras.io), scikit learn 
* Database: PostgreSQL 
```
pip install -U pip
pip install numpy scikit-learn pandas jupyter 
pip3 install torch torchvision 

``` 
#### Project scope  
Data type: time series numerical data (the sensor measured at the 100 Hz= 100 data points/s)   
Features: gyro-XYZ, lowAccel-XYZ, highAccel-XYZ  
Label: exercise categories (around 15 different exercise)  
Problem type: Human Activity Recognition 

#### Pipeline: 
* Data preprocessing:
   * Apply SVD and PCA to extract key features varied on specific exercise type 
   * Data normalization to get mean, std; using minmax and FFT, FIR filtering to get more data variants during repeated time chunks  
* Algorithm: 
   * Build the classifier model, split data into train_validation_test, train the model by logistic regression to implement the accuracy;   aim to try SVM for multiclass processing, at the same time will apply MLP, the simplified neural networks to run the model once we have much more datasets 
* Implementation and testing (TBD)   

#### Overview  
The final goal is to classifier the exercise to score the performance, CoderBunker colearners hypothesize to use the binary classifier to run the testing code, labeling the good performance one as 1, the rest is 0 in the initial stage; for the more datasets provided, we will use neural network to set 10 more different neurons for the probability testing. By the research we have done, we aim to transfer learning from UCI HAR datasets once we expand data features (min, max, entropy, correlation and etc), to implement model accuracy and flexibility.  

#### Status and Challenge 
We tested with time series analysis and plotly to visualize the data points, to segment the repeated pattern (a recurring pattern over a fixed period of time), which helps us to extract and filter features based on the graph moves distance. 

For more experimental features variants, next step we will manually extract repeated pattern from the graph, the aim is to retrieve a set of data with high correlation, allowing us to extract the best candidates for the training dataset. Breaking down two steps by project objective  
- Count how many exercise did (count the peaks) 
- Detect the graph moves by pattern shapes 

 #### Contributing :monkey: 
- Anne-Sophie, Jacob works for the data preprocessing and build the database in Postgres, currently Sophie is doing the structured data labelling 
- Tanmay, Chloe and Chris works for test with those training data with algorithm and assist the data preprocessing part, testing and validating the methodology that applicable to this exercise classification problem 
- Ricky and Istus suggested technical approaches and guide the team based on project timeline and scope 

#### Related working reference: 
- CodeBook UCI Data Analysis. (n.d.). Retrieved July 23, 2018, from http://ucihar-data-analysis.readthedocs.io/en/latest/CodeBook/
- Paul, J. (n.d.). Peak signal detection in real time time series data. Retrieved July 23, 2018, from https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/22640362#22640362
- Guidance and help from Ray, Sean, Mohammed from CoderBunker Community  
