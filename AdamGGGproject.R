#CODE FOR KAGGLE GHOULS,GOBLINS,GHOST

library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
# read in data
train <- vroom("train.csv")
test <- vroom("test.csv")
imputed_set <- vroom("trainimputedset.csv")

#DATA IMPUTATION for missing values
imputation_recipe <- recipe(type~., data=imputed_set) %>%
  step_impute_mean(all_numeric_predictors())

prep <- prep(imputation_recipe)
baked <- bake(prep, new_data = imputed_set)

rmse_vec(train[is.na(imputed_set)], baked[is.na(imputed_set)])

KNNimputationrecipe <-
  recipe(type~., data=imputed_set) %>%
  step_impute_knn(bone_length, impute_with = imp_vars(c('has_soul')), neighbors = 3) %>%
  step_impute_knn(rotting_flesh, impute_with = imp_vars(c('has_soul', 'bone_length')), neighbors = 3) %>%
  step_impute_knn(hair_length, impute_with = imp_vars(c('has_soul', 'bone_length', 'rotting_flesh')), neighbors = 3)

prep <- prep(KNNimputationrecipe)
bakedrecipe <- bake(prep, new_data = imputed_set)

rmse_vec(train[is.na(imputed_set)], baked[is.na(imputed_set)])


linearImputationRecipe <-
  recipe(type~., data=imputed_set) %>%
  step_impute_linear(bone_length, impute_with = c('has_soul', 'type')) %>%
  step_impute_linear(rotting_flesh, impute_with = c('has_soul', 'type', 'bone_length')) %>%
  step_impute_linear(hair_length,
                     impute_with = c('has_soul', 'type', 'bone_length', 'rotting_flesh'))


prepLinear <- prep(linearImputationRecipe)
bakedLinear <- bake(prepLinear, new_data = imputed_set)

rmse_vec(train[is.na(imputed_set)], baked[is.na(imputed_set)])
# best imputation RMSE: Linear, 0.1312063

bagimputationreceta <-
  recipe(type~., data=imputed_set) %>%
  step_impute_bag(bone_length, impute_with = c('has_soul', 'type'), trees = 500) %>%
  step_impute_bag(rotting_flesh, impute_with = c('has_soul', 'type', 'bone_length'), trees = 500) %>%
  step_impute_bag(hair_length,
                  impute_with = c('has_soul', 'type', 'bone_length', 'rotting_flesh'), trees = 500)



prep <- prep(bagimputationreceta)
baked <- bake(prep, new_data = imputed_set)

rmse_vec(train[is.na(imputed_set)], baked[is.na(imputed_set)])





################ now time for modeling
#----RANDOM FOREST-----#

randomForestRecipe <-
  recipe(type~., data=train) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_pca(all_predictors(), threshold = .8)

rrandomForestModelGGG <- rand_forest(min_n = tune(), mtry = tune()) %>%
  set_engine('ranger') %>%
  set_mode('classification')


randomForestGGG_wf <-
  workflow() %>%
  add_recipe(randomForestRecipe) %>%
  add_model(rrandomForestModelGGG)




#set up a tuning grid
tuning_grid <-
  grid_regular(mtry(range = c(1,10)),
               min_n(range(1,30)),
               levels = 15)

#split into folds
folds <- vfold_cv(train, v = 5, repeats = 1)

#run cv

CV_results <-
  randomForestGGG_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# find best tuning parm values

best_tuneRF <-
  CV_results %>%
  select_best("accuracy")

#finalize wf and get preds

final_RFGGG_wf <-
  randomForestGGG_wf %>%
  finalize_workflow(best_tuneRF) %>%
  fit(data = train)

rf_preds <-
  final_wf %>%
  predict(new_data = test, type = 'class')

#prepare and export preds to csv for kaggle

randomforestpreds <- tibble(id = test$id, type = rf_preds$.pred_class)


vroom_write(randomforestpreds, "RandomForestGGG", delim = ",")


#----KNN----#

knn_recipe <-
  recipe(type~., data=train) %>%
  step_normalize(all_numeric_predictors())

knn_mod <- nearest_neighbor(neighbors = tune(), dist_power = tune()) %>%
  set_engine('kknn') %>%
  set_mode('classification')


knn_wf <-
  workflow() %>%
  add_recipe(knn_recipe) %>%
  add_model(knn_mod)


## set up a tuning grid
tuning_grid <-
  grid_regular(neighbors(),
               dist_power(),
               levels = 6)

## split into folds
folds <- vfold_cv(train, v = 5, repeats = 1)

# run cv

CV_results <-
  knn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("accuracy")

# finalize wf and get preds

final_wf <-
  knn_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = train)

knn_preds <-
  final_wf %>%
  predict(new_data = test, type = 'class')

# prepare and export preds to csv for kaggle

knn_output <- tibble(id = test$id, type = knn_preds$.pred_class)

vroom_write(knn_output, "knnGGG.csv", delim = ",")


#-----NAIVE BAYES-----#
library(discrim)

nb_recipe <-
  recipe(type~., data=train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_predictors(), threshold = .99)

nb_mod <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_engine('naivebayes') %>%
  set_mode('classification')


nb_wf <-
  workflow() %>%
  add_recipe(nb_recipe) %>%
  add_model(nb_mod)


## set up a tuning grid
tuning_grid <-
  grid_regular(Laplace(),
               smoothness(),
               levels = 6)

## split into folds
folds <- vfold_cv(train, v = 5, repeats = 1)

# run cv

CV_results <-
  nb_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("accuracy")

# finalize wf and get preds

final_wf <-
  nb_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = train)

nb_preds <-
  final_wf %>%
  predict(new_data = test, type = 'class')

# prepare and export preds to csv for kaggle

nb_output <- tibble(id = test$id, type = nb_preds$.pred_class)

vroom_write(nb_output, "naivebayesGGG.csv", delim = ",")






#-----Support Vector Machines-----#


svm_recipe <- recipe(type~., data=train) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())

svmRad <- svm_rbf(rbf_sigma=tune(), cost = tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- 
  workflow() %>%
  add_recipe(svm_recipe) %>%
  add_model(svmRad)

## set up a tuning grid
tuning_grid <-
  grid_regular(rbf_sigma(),
               cost(),
               levels = 10)

#split into folds
folds <- vfold_cv(train, v = 5, repeats = 1)

#run cv

CV_results <-
  svm_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

#find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("accuracy")

# finalize wf and get preds

final_wf <-
  svm_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = train)

svm_preds <-
  final_wf %>%
  predict(new_data = test, type = "class")

# prepare and export preds to csv for kaggle

svm_output <- tibble(id = test$id, type = svm_preds$.pred_class)


vroom_write(svm_output, "supportvectorpredsGGG.csv", delim = ",")



#-----BOOSTING-----#
library(bonsai)
library(lightgbm)
boostingReceta <- recipe(type~., data=train) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())

boost_mod <- boost_tree(trees= 500, tree_depth = 1,
                        learn_rate = .001) %>% # set or tune
  set_mode("classification") %>%
  set_engine("lightgbm")

boost_wf <- 
  workflow() %>%
  add_recipe(boostingReceta) %>%
  add_model(boost_mod)

#set up a tuning grid
tuning_grid <-
  grid_regular(trees(),
               tree_depth(),
               learn_rate(),
               levels = 2)

#split into folds
folds <- vfold_cv(train, v = 4, repeats = 1)

# run cv
CV_results <-
  boost_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

#find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("accuracy")

# finalize wf and get preds

final_wf <-
  boost_wf %>%
  #finalize_workflow(best_tune) %>%
  fit(data = train)

boost_preds <-
  final_wf %>%
  predict(new_data = test, type = "class")

#prepare and export preds to csv for kaggle

boost_output <- tibble(id = test$id, type = boost_preds$.pred_class)


vroom_write(boost_output, "boostedGGGpredictions.csv", delim = ",")



#-----NEURAL NETWORK-----#

nn_recipe <- recipe(type~., data = train) %>%
  update_role(id, new_role="id") %>%
  step_mutate(color = as.factor(color)) %>% ## Turn color to factor then dummy encode color
  step_dummy(color) %>%
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

nn_model <- mlp(hidden_units = 3,
                epochs = 250) %>%
  set_engine("nnet") %>% #verbose = 0 prints off less
  set_mode("classification")
nn_wf <- 
  workflow() %>%
  add_model(nn_model) %>%
  add_recipe(nn_recipe)
nn_tuning_grid <- grid_regular(hidden_units(range=c(1, 60)),
                               levels=60)
## split into folds
folds <- vfold_cv(train, v = 5, repeats = 1)

# run cv
CV_results <-
  nn_wf %>%
  tune_grid(resamples = folds,
            grid = nn_tuning_grid,
            metrics = metric_set(accuracy))

best_tune <-
  CV_results %>%
  select_best("accuracy")



# finalize wf and get preds
final_wf <-
  nn_wf %>%
  #finalize_workflow(best_tune) %>%
  fit(data = train)

nn_preds <-
  final_wf %>%
  predict(new_data = test, type = "class")

# prepare and export preds to csv for kaggle

nn_output <- tibble(id = test$id, type = nn_preds$.pred_class)


vroom_write(nn_output, "neuralnetworkGGG.csv", delim = ",")



tuned_nn <-
  nn_wf %>%
  tune_grid(resamples = folds,
            grid = nn_tuning_grid,
            metrics = metric_set(accuracy))
