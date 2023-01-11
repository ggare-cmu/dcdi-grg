import numpy as np

import torch

import os

import zero

# import sys
# sys.path.append('./')

import utils


import cascade_train

# Runtime optimization of sklearn uisng Intel - Ref: https://github.com/intel/scikit-learn-intelex
from sklearnex import patch_sklearn, config_context
patch_sklearn()

# with config_context(target_offload="gpu:0"):
#     clustering = DBSCAN(eps=3, min_samples=2).fit(X)


from sklearn import tree, svm, neighbors
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor

def fitLargeMLP(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, 
        label_classes, feature_type = '',
        n_trial = 3, hidden_layer_sizes = (128, 64, 32), max_iter = 200, verbose = True, USE_GPU = False):

    best_clf = None
    best_acc = -np.inf
    for idx in range(n_trial):

        # clf = MLPClassifier()

        if feature_type == "continous":
            clf = MLPRegressor(
                # hidden_layer_sizes = (128, 64),
                # hidden_layer_sizes = (128, 64, 32),
                hidden_layer_sizes = hidden_layer_sizes,
                learning_rate = "adaptive", #constant
                max_iter = max_iter,
                verbose = verbose,
            )
        else:
            clf = MLPClassifier(
                # hidden_layer_sizes = (128, 64),
                # hidden_layer_sizes = (128, 64, 32),
                hidden_layer_sizes = hidden_layer_sizes,
                learning_rate = "adaptive", #constant
                max_iter = max_iter,
                verbose = verbose,
            )
        
        if USE_GPU:
            with config_context(target_offload="gpu:0"):
                clf = clf.fit(train_label_ft, gt_train_scores)
        else:
            clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        # #Map predictions to proper class labels; Here some class values can be missing
        # ml_predictions = label_classes[ml_predictions]

        # if feature_type == "continous":
        #     accuracy = clf.score(test_label_ft, gt_test_scores)
        # else:
        #     accuracy = (ml_predictions == gt_test_scores).mean()
        accuracy = clf.score(test_label_ft, gt_test_scores)
        print(f'[Trial-{idx}] ML model (MLP Large) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf

    if feature_type == "continous":
        ml_predictions = clf.predict(test_label_ft)
        ml_prob_predictions = ml_predictions
    else:
        ml_predictions = clf.predict(test_label_ft)
        ml_prob_predictions = clf.predict_proba(test_label_ft)


    # #Map predictions to proper class labels; Here some class values can be missing
    # ml_predictions = label_classes[ml_predictions]

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'ML model (MLP Large) accuracy = {accuracy}')

    
    return clf, accuracy, ml_predictions, ml_prob_predictions




def fitRandomForest(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, label_classes, n_trial = 3, USE_GPU = False):

    best_clf = None
    best_acc = -1
    for idx in range(n_trial):

        clf = RandomForestClassifier(n_estimators = 100) #The number of trees in the forest (default 100).
        
        if USE_GPU:
            with config_context(target_offload="gpu:0"):
                clf = clf.fit(train_label_ft, gt_train_scores)
        else:
            clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        accuracy = (ml_predictions == gt_test_scores).mean()
        print(f'[Trial-{idx}] ML model (RandomForest) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf
    ml_predictions = clf.predict(test_label_ft)
    ml_prob_predictions = clf.predict_proba(test_label_ft)

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'ML model (RandomForest) accuracy = {accuracy}')

    
    return clf, accuracy, ml_predictions, ml_prob_predictions



def fitDecisionTree(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, label_classes, n_trial = 3, USE_GPU = False):

    best_clf = None
    best_acc = -1
    for idx in range(n_trial):

        clf = tree.DecisionTreeClassifier()
        
        if USE_GPU:
            with config_context(target_offload="gpu:0"):
                clf = clf.fit(train_label_ft, gt_train_scores)
        else:
            clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        accuracy = (ml_predictions == gt_test_scores).mean()
        print(f'[Trial-{idx}] ML model (DecisionTree) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf
    ml_predictions = clf.predict(test_label_ft)
    ml_prob_predictions = clf.predict_proba(test_label_ft)

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'ML model (DecisionTree) accuracy = {accuracy}')


    
    return clf, accuracy, ml_predictions, ml_prob_predictions



def fitSVM(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, label_classes, n_trial = 3, USE_GPU = False):

    best_clf = None
    best_acc = -1
    for idx in range(n_trial):

        # clf = svm.SVC()
        clf = svm.SVC(probability = True) #Enable probability predictions
        
        if USE_GPU:
            with config_context(target_offload="gpu:0"):
                clf = clf.fit(train_label_ft, gt_train_scores)
        else:
            clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        accuracy = (ml_predictions == gt_test_scores).mean()
        print(f'[Trial-{idx}] ML model (SVM) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf
    ml_predictions = clf.predict(test_label_ft)
    ml_prob_predictions = clf.predict_proba(test_label_ft)

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'ML model (SVM) accuracy = {accuracy}')

    
    return clf, accuracy, ml_predictions, ml_prob_predictions




#Numpy new way to sample from distributions: Ref: https://numpy.org/doc/stable/reference/random/index.html#random-quick-start
from numpy.random import default_rng
rng = default_rng()




from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, average_precision_score, multilabel_confusion_matrix
from scipy.special import softmax
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def calScores(preds, prob_preds, targets, class_names, task, logger, binary_cross_entropy = False, skip_auc = False):

    labels = np.arange(len(class_names))
    

    accuracy = accuracy_score(targets, preds)

    if binary_cross_entropy:
        confusionMatrix = multilabel_confusion_matrix(targets, preds, labels = labels)
    else:
        confusionMatrix = confusion_matrix(targets, preds, labels = labels)
    
    # confusionMatrix = confusion_matrix(targets, preds, labels = labels)
    
    if binary_cross_entropy or skip_auc:
        # auc = "-"
        
        if len(prob_preds.shape) == 1 or prob_preds.shape[1] == 1:
            auc = roc_auc_score(targets, prob_preds[:,0])
        else:
            
            assert np.all(np.isclose(np.unique(prob_preds.sum(1)),1)), "Error! Probability does not sum to one."

            auc = roc_auc_score(targets, prob_preds[:,1])
    else:
        auc = roc_auc_score(targets, prob_preds, average = "weighted", multi_class = "ovo") # multi_class = "ovr"
    precision = precision_score(targets, preds, average='weighted') #score-All average
    recall = recall_score(targets, preds, average='weighted') #score-All average
    f1 = f1_score(targets, preds, average='weighted') #score-All average
        
    classificationReport = classification_report(targets, preds, labels = labels, target_names = class_names, digits=5)

    logger.log(f"auc = {auc}")
    logger.log(f"accuracy = {accuracy}")
    logger.log(f"precision = {precision}")
    logger.log(f"recall = {recall}")
    logger.log(f"f1 = {f1}")
    logger.log(f"confusionMatrix = \n {confusionMatrix}")
    logger.log(f"classificationReport = \n {classificationReport}")


    results_dict = {}
    results_dict["auc"] = auc
    results_dict["accuracy"] = accuracy
    results_dict["precision"] = precision
    results_dict["recall"] = recall
    results_dict["f1"] = f1
    results_dict["confusionMatrix"] = confusionMatrix.tolist()
    results_dict["classificationReport"] = classificationReport

    return results_dict


def upsampleFeatures(labels, features):

    classes, count = np.unique(labels, return_counts = True)
    print(f"[Pre-Upsampling] classes, count = {classes, count}")   
    
    max_count = max(count)

    label_indices = []
    for c in classes:

        c_idx = np.where(labels == c)[0]
        assert np.unique(labels[c_idx]) == c, "Error! Wrong class index filtered."

        #Bug-GRG : Since we sample randomly some of the videos are never sampled/included. 
        # So, make sure to only sample additional required videos after including all videos at least once!
        #For the max count class, set replace to False as setting it True might exclude some samples from training
        # upsample_c_idx = np.random.choice(c_idx, size = max_count, replace = len(c_idx) < max_count)
        if len(c_idx) < max_count:
            # upsample_c_idx = np.array(c_idx.tolist() + np.random.choice(c_idx, size = max_count - len(c_idx), replace = len(c_idx) < max_count).tolist())
            upsample_c_idx = np.array(c_idx.tolist() + np.random.choice(c_idx, size = max_count - len(c_idx), replace = max_count > 2*len(c_idx)).tolist())
        else:
            upsample_c_idx = c_idx
        
        np.random.shuffle(upsample_c_idx)
        
        assert c_idx.shape == np.unique(upsample_c_idx).shape, "Error! Some videos where excluded on updampling."

        label_indices.extend(upsample_c_idx)

    assert len(label_indices) == max_count * len(classes)

    upsample_label_indices = label_indices

    # upsampled_features = features[label_indices, :]
    upsampled_features = features[label_indices]

    upsampled_labels = labels[label_indices]

    classes, count = np.unique(upsampled_labels, return_counts = True)
    print(f"[Post-Upsampling] classes, count = {classes, count}")   

    assert np.array_equal(count, max_count * np.ones(len(classes))), "Error! Upsampling didn't result in class-balance"

    return upsampled_labels, upsampled_features, upsample_label_indices



def main(class_parents, class_parents_idx, seed,
        causal_discovery_exp_dir, task, dataset_path,
        exp_name = "Classifer_Causal_parents",
        trial = 'T1',
    ):

    #Set seed to improve reproducibility 
    zero.improve_reproducibility(seed)


    max_iter = 200 #200


    print(f"Downstream classification for task {task} trial {trial}")
    
    results_dir = os.path.join(causal_discovery_exp_dir, f"{exp_name}_{trial}")
    utils.createDirIfDoesntExists(results_dir)

   
   
        
    ### Load training data
    
    train_data_path = f"{dataset_path}/{task.split('_')[0]}_TrainUpsampledPatient.npz"

    train_dataset = np.load(train_data_path, allow_pickle = True)

    train_data = train_dataset['data_obs']


    vars_list = train_dataset['vars_list']
    class_list = train_dataset['class_list']

    feature_type_dict = train_dataset['feature_type'].item()

    features_list = vars_list.tolist() + class_list.tolist()
    


    node_names_mapping = {}
    for idx, var in enumerate(features_list):
        # node_names_mapping[f"$X_\\{{idx}\\}$"] = var
        node_names_mapping["$X_{" + str(idx+1) + "}$"] = var

    causal_feature_mapping = {}
    for idx, var in enumerate(features_list):
        causal_feature_mapping[var] = idx

    num_vars = len(vars_list)
    num_classes = len(class_list)

    binary_cross_entropy = num_classes == 2


    class_names = class_list


    category_classes = list(range(num_vars, num_vars+num_classes))



    #Logger init 
    report_path = os.path.join(results_dir, f"classification_report_{task}.txt")
    logger = utils.Logger(report_path)

    logger.log(f"Classification report")

    logger.log(f"Exp name: {exp_name}")



   
    ## Test the synthetic data

    print(f"train_data.shape = {train_data.shape}")

    train_features = train_data[:, :num_vars]

    train_labels = train_data[:, category_classes]

    UpSampleData = True #False
    if UpSampleData:
        upsam_train_labels, upsam_train_features, upsample_label_indices = upsampleFeatures(labels = train_labels.argmax(1), features = train_features) 
        train_labels = train_labels[upsample_label_indices]
        train_features = upsam_train_features

        assert train_labels.shape[0] == train_features.shape[0], "Error! Upsampled labels and feature count does not match."

    print(f"train_labels.shape = {train_labels.shape}")
    print(f"train_features.shape = {train_features.shape}")


    causal_train_features = train_features[:, class_parents_idx]
    assert causal_train_features.shape == (train_features.shape[0], len(class_parents)), "Error! Wrong causal parents filtered."

    logger.log(f"Causal parents [len = {len(class_parents)}] = {class_parents}")

    test_data_path = f"{dataset_path}/{task.split('_')[0]}_TestUpsampledPatient.npz"

    test_data = np.load(test_data_path)['data_obs']


    val_data_path = f"{dataset_path}/{task.split('_')[0]}_ValUpsampledPatient.npz"

    val_data = np.load(val_data_path)['data_obs']


    test_features = test_data[:, :num_vars]

    test_labels = test_data[:, category_classes]

    causal_test_features = test_features[:, class_parents_idx]
    assert causal_test_features.shape == (test_features.shape[0], len(class_parents)), "Error! Wrong causal parents filtered."

    val_features = val_data[:, :num_vars]

    val_labels = val_data[:, category_classes]

    causal_val_features = val_features[:, class_parents_idx]
    assert causal_val_features.shape == (val_features.shape[0], len(class_parents)), "Error! Wrong causal parents filtered."

    label_classes = np.unique(train_labels.argmax(1))



    ### Evaluate the synthetic data on Test set









    ##Train model
    model, accuracy, ml_predictions, ml_prob_predictions = fitLargeMLP(
                                train_label_ft =  train_features, test_label_ft = test_features, 
                                # gt_train_scores = train_labels, gt_test_scores = test_labels, 
                                gt_train_scores = train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                label_classes = label_classes,
                                # features_type = "categorical",
                                # n_trial = 3, hidden_layer_sizes = (128, 64, 32),
                                n_trial = 3, hidden_layer_sizes = (100),
                                max_iter = max_iter,
                                verbose = False
                            )

    logger.log(f"[MLPlarge] Accuracy on original train set = {accuracy}")

    # model_results_dict = calScores(preds = ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger)

    model_results_dict = calScores(preds = ml_predictions, prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), 
                targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger,
                binary_cross_entropy = binary_cross_entropy)
    


    ##Train model
    cu_model, cu_accuracy, cu_ml_predictions, cu_ml_prob_predictions = fitLargeMLP(
                                train_label_ft =  causal_train_features, test_label_ft = causal_test_features, 
                                # gt_train_scores = train_labels, gt_test_scores = test_labels, 
                                gt_train_scores = train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                label_classes = label_classes,
                                # features_type = "categorical",
                                # n_trial = 3, hidden_layer_sizes = (128, 64, 32),
                                n_trial = 3, hidden_layer_sizes = (100),
                                max_iter = max_iter,
                                verbose = False
                            )

    logger.log(f"[MLPlarge] Accuracy on causal original train set = {cu_accuracy}")

    # model_results_dict = calScores(preds = ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger)

    cu_model_results_dict = calScores(preds = cu_ml_predictions, prob_preds = torch.softmax(torch.Tensor(cu_ml_prob_predictions), dim = 1).numpy(), 
                targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger,
                binary_cross_entropy = binary_cross_entropy)






    ######### Custom MLP ############

    ##Train model - baseline 
    cm_model, cm_accuracy, cm_ml_predictions, cm_ml_prob_predictions = cascade_train.fitCustomMLP(
                                feat_train = train_features, target_train = train_labels.argmax(1),
                                feat_test = test_features, target_test = test_labels.argmax(1), 
                                label_classes = label_classes,
                                # n_trial = 3, hidden_layer_sizes = (100),
                                train_epochs = max_iter,
                                seed = seed,
                            )

    logger.log(f"[Custom MLP] Accuracy on original train set = {cm_accuracy}")

    # model_results_dict = calScores(preds = ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger)

    cm_model_results_dict = calScores(preds = cm_ml_predictions, prob_preds = torch.softmax(torch.Tensor(cm_ml_prob_predictions), dim = 1).numpy() if not binary_cross_entropy else torch.sigmoid(torch.tensor(cm_ml_prob_predictions)).numpy(), 
                targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger,
                binary_cross_entropy = binary_cross_entropy)



    ##Train model - baseline l1 regularized
    cm_l1_model, cm_l1_accuracy, cm_l1_ml_predictions, cm_l1_ml_prob_predictions = cascade_train.fitCustomMLP(
                                feat_train = train_features, target_train = train_labels.argmax(1),
                                feat_test = test_features, target_test = test_labels.argmax(1), 
                                label_classes = label_classes,
                                # n_trial = 3, hidden_layer_sizes = (100),
                                train_epochs = max_iter,
                                l1_regularize = True,
                                seed = seed,
                            )

    logger.log(f"[Custom MLP] Accuracy on L1 regularized original train set = {cm_l1_accuracy}")

    # model_results_dict = calScores(preds = ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger)

    cm_l1_model_results_dict = calScores(preds = cm_l1_ml_predictions, prob_preds = torch.softmax(torch.Tensor(cm_l1_ml_prob_predictions), dim = 1).numpy() if not binary_cross_entropy else torch.sigmoid(torch.tensor(cm_l1_ml_prob_predictions)).numpy(), 
                targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger,
                binary_cross_entropy = binary_cross_entropy)
    


    ##Train model - causal_parents  
    cm_cu_model, cm_cu_accuracy, cm_cu_ml_predictions, cm_cu_ml_prob_predictions = cascade_train.fitCustomMLP(
                                feat_train = causal_train_features, target_train = train_labels.argmax(1),
                                feat_test = causal_test_features, target_test = test_labels.argmax(1), 
                                label_classes = label_classes,
                                # n_trial = 3, hidden_layer_sizes = (100),
                                train_epochs = max_iter,
                                seed = seed,
                            )

    logger.log(f"[Custom MLP] Accuracy on causal original train set = {cm_cu_accuracy}")

    # model_results_dict = calScores(preds = ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger)

    cm_cu_model_results_dict = calScores(preds = cm_cu_ml_predictions, prob_preds = torch.softmax(torch.Tensor(cm_cu_ml_prob_predictions), dim = 1).numpy() if not binary_cross_entropy else torch.sigmoid(torch.tensor(cm_cu_ml_prob_predictions)).numpy(), 
                targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger,
                binary_cross_entropy = binary_cross_entropy)
    


    logger.log(f"\n\n*** Custom MLP (start) ***\n\n")

    logger.log(f"Task = {task} | Causal DAG")
    logger.log(f"*** Accuracy metric ***")
    logger.log(f"Trial = {trial}")
    logger.log(f"[Custom MLP] Accuracy on original train set = {cm_accuracy}")
    logger.log(f"[Custom MLP] Accuracy on L1 regularized original train set = {cm_l1_accuracy}")
    logger.log(f"[Custom MLP] Accuracy on casual original train set = {cm_cu_accuracy}")

    logger.log(f"*** AUC metric ***")
    logger.log(f"[Custom MLP] AUC on original train set = {cm_model_results_dict['auc']}")
    logger.log(f"[Custom MLP] AUC on L1 regularized original train set = {cm_l1_model_results_dict['auc']}")
    logger.log(f"[Custom MLP] AUC on causal original train set = {cm_cu_model_results_dict['auc']}")
  
    logger.log(f"\n\n*** Custom MLP (end) ***\n\n")

    ######### Random Forest ############

    runRandomForest = True
    if runRandomForest:

        ##Train model
        rf_model, rf_accuracy, rf_ml_predictions, rf_ml_prob_predictions = fitRandomForest(
                                    train_label_ft =  train_features, test_label_ft = test_features, 
                                    # gt_train_scores = train_labels, gt_test_scores = test_labels, 
                                    gt_train_scores = train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                    label_classes = label_classes,
                                    n_trial = 3
                                )

        logger.log(f"[RandomForest] Accuracy on original train set = {rf_accuracy}")

        # model_results_dict = calScores(preds = ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger)

        rf_model_results_dict = calScores(preds = rf_ml_predictions, prob_preds = torch.softmax(torch.Tensor(rf_ml_prob_predictions), dim = 1).numpy(), 
                    targets = test_labels.argmax(1), class_names = class_names, task = task+"-rf", logger = logger,
                    binary_cross_entropy = binary_cross_entropy)




        ##Train model
        cu_rf_model, cu_rf_accuracy, cu_rf_ml_predictions, cu_rf_ml_prob_predictions = fitRandomForest(
                                    train_label_ft =  causal_train_features, test_label_ft = causal_test_features, 
                                    # gt_train_scores = train_labels, gt_test_scores = test_labels, 
                                    gt_train_scores = train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                    label_classes = label_classes,
                                    n_trial = 3
                                )

        logger.log(f"[RandomForest] Accuracy on causal original train set = {cu_rf_accuracy}")

        # model_results_dict = calScores(preds = ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger)

        cu_rf_model_results_dict = calScores(preds = cu_rf_ml_predictions, prob_preds = torch.softmax(torch.Tensor(cu_rf_ml_prob_predictions), dim = 1).numpy(), 
                    targets = test_labels.argmax(1), class_names = class_names, task = task+"-rf", logger = logger,
                    binary_cross_entropy = binary_cross_entropy)


    ######### DecisionTree ############

    runDT = True
    if runDT:

        ##Train model
        dt_model, dt_accuracy, dt_ml_predictions, dt_ml_prob_predictions = fitDecisionTree(
                                    train_label_ft =  train_features, test_label_ft = test_features, 
                                    # gt_train_scores = train_labels, gt_test_scores = test_labels, 
                                    gt_train_scores = train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                    label_classes = label_classes,
                                    n_trial = 3
                                )

        logger.log(f"[DT] Accuracy on original train set = {dt_accuracy}")

        # model_results_dict = calScores(preds = ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger)

        dt_model_results_dict = calScores(preds = dt_ml_predictions, prob_preds = torch.softmax(torch.Tensor(dt_ml_prob_predictions), dim = 1).numpy(), 
                    targets = test_labels.argmax(1), class_names = class_names, task = task+"-svm", logger = logger,
                    binary_cross_entropy = binary_cross_entropy)


        ##Train model
        cu_dt_model, cu_dt_accuracy, cu_dt_ml_predictions, cu_dt_ml_prob_predictions = fitDecisionTree(
                                    train_label_ft =  causal_train_features, test_label_ft = causal_test_features, 
                                    # gt_train_scores = train_labels, gt_test_scores = test_labels, 
                                    gt_train_scores = train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                    label_classes = label_classes,
                                    n_trial = 3
                                )

        logger.log(f"[DT] Accuracy on causal original train set = {cu_dt_accuracy}")

        # model_results_dict = calScores(preds = ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger)

        cu_dt_model_results_dict = calScores(preds = cu_dt_ml_predictions, prob_preds = torch.softmax(torch.Tensor(cu_dt_ml_prob_predictions), dim = 1).numpy(), 
                    targets = test_labels.argmax(1), class_names = class_names, task = task+"-svm", logger = logger,
                    binary_cross_entropy = binary_cross_entropy)

        



    logger.log(f"Causal parents [len = {len(class_parents)}] = {class_parents}")

    logger.log(f"Task = {task} | Causal DAG")
    logger.log(f"Trial = {trial}")
    logger.log(f"[MLPlarge] Accuracy on original train set = {accuracy}")
    logger.log(f"[MLPlarge] Accuracy on casual original train set = {cu_accuracy}")
    logger.log(f"[RandomForest] Accuracy on original train set = {rf_accuracy}")
    logger.log(f"[RandomForest] Accuracy on causal original train set = {cu_rf_accuracy}")
    logger.log(f"[DT] Accuracy on original train set = {dt_accuracy}")
    logger.log(f"[DT] Accuracy on causal original train set = {cu_dt_accuracy}")

    # logger.log(f"{accuracy} \n{rf_accuracy} \n{dt_accuracy} \n{only_synthetic_accuracy} \n{rf_only_synthetic_accuracy} \n{dt_only_synthetic_accuracy} \n{synthetic_accuracy} \n{rf_synthetic_accuracy} \n{dt_synthetic_accuracy}")
    


    logger.log(f"*** AUC metric ***")
    logger.log(f"Task = {task} | Causal DAG")
    logger.log(f"Trial = {trial}")
    logger.log(f"[MLPlarge] AUC on original train set = {model_results_dict['auc']}")
    logger.log(f"[MLPlarge] AUC on causal original train set = {cu_model_results_dict['auc']}")
    logger.log(f"[RandomForest] AUC on original train set = {rf_model_results_dict['auc']}")
    logger.log(f"[RandomForest] AUC on causal original train set = {cu_rf_model_results_dict['auc']}")
    logger.log(f"[DT] AUC on original train set = {dt_model_results_dict['auc']}")
    logger.log(f"[DT] AUC on causal original train set = {cu_dt_model_results_dict['auc']}")

    # logger.log(f"{model_results_dict['auc']} \n{rf_model_results_dict['auc']} \n{dt_model_results_dict['auc']} \n{only_synthetic_model_results_dict['auc']} \n{rf_only_synthetic_model_results_dict['auc']} \n{dt_only_synthetic_model_results_dict['auc']} \n{synthetic_model_results_dict['auc']} \n{rf_synthetic_model_results_dict['auc']} \n{dt_synthetic_model_results_dict['auc']}")
    


    logger.close()

    pass





if __name__ == "__main__":
    print("Started...")

    main()
    
    print("Finished!")