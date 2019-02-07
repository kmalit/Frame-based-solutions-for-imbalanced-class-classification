# Frame-based-solutions-for-imbalanced-class-classification
## The Research Question: The effect on precision and recall of frame-based sampling in biased class(es) classification problems.
### Optimal Experiment Design: 
	* Restricting the training examples for the heavy class(es) to the instances lying on the border of the convex hull of all these data points – The Frame
### Does Frame Based sampling of the heavier class(es):
	* Preserve/improve the precision score of the heavy class(es)?
	* Improve precision score of the smaller class(es) ?
	* Improve Recall of the smaller class(es)?
	* How do the results compare to the boosting methods?
### Methodology
  * Imbalance data: One class has a lot more observations than the other class
   ![alt text](https://github.com/kmalit/Frame-based-solutions-for-imbalanced-class-classification/blob/master/data/plots/imbalanced_class.png)
  #### Method 1: The full frame:
   ![alt text](https://github.com/kmalit/Frame-based-solutions-for-imbalanced-class-classification/blob/master/data/plots/the_frame.png)
    * Compute the overall frame of the data
    * Sample training data from the computed frame
    * Fit a classifier on the train data sampled and test results on the remaining data
  #### Method 2: Individual class frames
   ![alt text](https://github.com/kmalit/Frame-based-solutions-for-imbalanced-class-classification/blob/master/data/plots/each_class_frame.png)
    * Compute the frames individually for each class of data
    * Sample train data from the computed frames
    * Train a classifier on the train set and test on the remaining data set.
  #### Experiment set up
    * Several linear and non linear classifiers were fitted on 9 different data sets to observe the trend and impact of frame based sampling
    * Datasets – 9:
        * N points: 5000 – 15000
        * Dimensions: 5 – 9
        * Class weights: [0.95,5], [0.85,0.15], [0.7,0.3] 

    * Models fitted
        * Logistic regression
        * Linear SVM
        * RF
        * Gradient Boosted Trees
