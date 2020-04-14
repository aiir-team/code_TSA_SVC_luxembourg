https://www.analyticsvidhya.com/blog/2018/07/hands-on-sentiment-analysis-dataset-python/

https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/#

https://towardsdatascience.com/creating-the-twitter-sentiment-analysis-program-in-python-with-naive-bayes-classification-672e5589a7ed

# Install
```code 
1. Open terminal on linux and do
    
    conda upgrade conda 
    conda create --name twitter
    conda activate twitter
    
    conda install pip
    conda install -c conda-forge mkl_fft
    conda install -c conda-forge wordcloud

    pip install -r requirements.txt
```

# Guide
```code 
1. utils file: including PreProcessing class (for twitter data) and HandlingIO class 
2. svm_model.py: for testing purpose 
3. All test1, test2, test3 using gridSearchCV, but the differents are:
    + test1: divided dataset into training and testing test with different ratio, then feeding training set into gridSearch
    + test2: divided dataset intro training and testing with equals ratio (in case 0.5:0.5 of test1).
    + test3: feed the whole dataset into gridSearchCV, let it do its job.

4. All test11, test12 using gridSearchCV, but the differents are the same above test. But 2 test for classifying 2 classes 
    (negative and positve), in above test we classifying 3 classes (positive, neutral and negative).

5. test22 for 2 classes, different dataset (~32k example)

```