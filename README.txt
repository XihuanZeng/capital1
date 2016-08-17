--------------------------------------------------------------------------------
|         Code Test Part 1: Model building on a synthetic dataset              | 
--------------------------------------------------------------------------------
please put codetest_train.txt and codetest_test.txt in the root dir(same as question1.py)

-prediction for test file: myprediction.txt

-method summary: use a weighted average of XGBoost Regressor, Random Forest Regressor, 
		 Ridge and Lasso

-estimated score(MSE on holdout set): 11.92

-implementation of my code: 
	- to train the model: 
		- python question1.py --mode train --cross-validation on --num-grid-search 20
		- python question1.py --mode train
	- to test the model:
		- python question1.py --mode test

model details:
	- preprocessing: 
		- one hot encoding: for categorical variables, one hot encoding is implemented. 
				    NaN in categorical variables are treated as another level 
				    of this variable.
		- missing value imputation: NaN in categorical variables are mentioned above, 
					    for numerical variables, use mean of the variable to 
					    impute the missing values.
	- training:
		- algorithms used: XGBoost Regressor, Random Forest Regressor, Ridge and Lasso.
		- algorithm tried but not used: K Nearest Neighbors, Linear Regression.
		- standardization: this is implemented to each column, each column thereafter 
				   has 0 mean and unit variance.
		- dimension reduction: PCA is implemented for each algorithm, the dimension is 
			 	       tuned respectively.
		- train, hold-out set split: 80-20 on training data
		- hyper-parameter tuning: for each model, there are two kinds of hyperparameters, 
				          1st is n_component in PCA, 2nd is the model parameters 
				          for each algorithm(such as learning rate in XGB), a 
					  grid search CV(random search CV on 60 grid points) is 
					  implemented to tune the hyper-parameters.
		- performance on hold-out set
			- XGBoost: 12.57
			- Ridge: 13.88
			- Lasso: 14.10
			- Random Forest: 14.44
			- Ensemble model(0.5*XGBoost+0.3*Rdige+0.1*Lasso+0.1*RandomForest): 11.92

	- testing: Scikit-learn Pipeline make sures preprocessing in test set is exactly same as 
		   in training set. I rebuild the model on whole training set use the best 
		   parameters chosen by grid search CV


model reusability: my model can also train on other dataset, where the 1st column is the target 
		   value and the remaining columns are either numeric or categorical features. 
		   It will find out categorical and numeric features respectively and apply different 
		   missing value imputation strategy.

--------------------------------------------------------------------------------
|                       Code Test Part 2: Baby Names!                          |
--------------------------------------------------------------------------------

- to open the ipython file, run in command line: 
	- jupyter notebook question2.ipynb, or
	- ipython notebook question2.ipynb



