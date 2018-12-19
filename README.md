# Personality-Attribution-using-Natural-Language-Processing

## A project by Chirayu Desai, Akhilesh Hegde and Yuzhou Yin

All files take one of the personalitiy types as command line argument to predict and assume that the data files required by them are in the same directory.

Example usage:  1) python file-name.py trait <br>
                trait can take vales form 0 to 4 based on the trait for which the user wants to run model denoted by:<br>
                0: Extraversion
                1: Neuroticism
                2: Agreeableness
                3: Conscientiousness
                4: Openness

The details of various ways and options of running each file can be found in the top document comment of each file.

essays.csv is our dataset file.


For Naive Bayes, the Parameters and features were decided by running naive_grid_search.py, naive_iterative_para_opt.py and naive_stopwords_stemmed.py for each trait.
naive_simple.py runs for each trait for optimal values of smoothning parameter to generate published results.

For Multilayer perceptron classifier the Parameters and features were decided by running mlp_baseline.py, mlp_grid_search.py, mlp_kfold.py, mlp_iterative.py for each trait.
The average results of multiple runs of mlp_simple.py produce the published results. The details of results of each run can be found in mlp_avg_res.xlsx
