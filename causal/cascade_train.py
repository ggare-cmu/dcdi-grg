import numpy as np
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import tqdm

import zero

import sys
sys.path.append("./code")
import utils


class CustomDataset(Dataset):

    # def __init__(self, filenames, inputs, targets):
    def __init__(self, inputs, targets):
        
        self.filenames = np.arange(targets.shape[0])
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        input = self.inputs[idx]
        target = self.targets[idx]

        input = torch.tensor(input)
        target = torch.tensor(target)
        
        input = input.float()

        return filename, input, target




def fitCustomMLP(feat_train, target_train, feat_test, target_test,
        label_classes,
        l1_regularize = False,
        train_epochs = 200, batch_size=256, seed = 42):

    #Set seed to improve reproducibility 
    zero.improve_reproducibility(seed)


    in_features = feat_train.shape[1]
    out_classes = len(label_classes)

    if out_classes == 2:
        out_classes = 1

    train_dataset = CustomDataset(inputs = feat_train, targets = target_train)
    test_dataset = CustomDataset(inputs = feat_test, targets = target_test)


    train_sampler = None
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=8, pin_memory=True, sampler=train_sampler)
    
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=8, pin_memory=True)

    if out_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.MSELoss()

    
    model = nn.Sequential(
                nn.Linear(in_features, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, out_classes)
            )

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    model.to("cuda:1")

    #Train the model
    model, train_acc = train(model, train_dataloader, optimizer, criterion,
                            num_epochs = train_epochs, 
                            l1_regularize = l1_regularize)

    print(f"\n [MLP] Train acc = {train_acc} \n")

    #Test the model
    target_test, test_acc = test(model, test_dataloader, criterion,
                         return_features = False, return_prob_preds=True)

    print(f"\n [MLP] Test acc = {test_acc} \n")

    return model, test_acc, (torch.sigmoid(torch.tensor(target_test)) > 0.5), target_test



def train(model, train_dataloader, optimizer, criterion,
            l1_regularize = False,
            device = "cuda:1", num_epochs = 100):

    model.to(device)

    print(f"l1_regularize = {l1_regularize}")
    
    # for epoch in range(1, num_epochs + 1):
    for epoch in tqdm.tqdm(range(1, num_epochs + 1)):
        
        # train_acc = torchmetrics.Accuracy(num_classes = num_classes, average = "weighted", multiclass = True)
        # train_acc.to(device)

        inputs_list = []
        targets_list = []
        preds_list = []
        
        model.train()

        # for idx, (filenames, inputs, targets) in enumerate(tqdm.tqdm(train_dataloader)):
        for idx, (filenames, inputs, targets) in enumerate(train_dataloader):

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = computeLoss(criterion, outputs, targets)

            if l1_regularize:
                # l2_loss = sum(p.pow(2.0).sum() for p in model.parameters())
                # l1_loss = sum(p.abs() for p in model.parameters()).sum()
                # l1_loss = sum(p.abs().sum() for p in model.parameters())
                # l1_loss = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)

                l1_loss = [p.abs().mean() for p in model.parameters() if p.requires_grad]
                l1_loss = sum(l1_loss)/len(l1_loss)

                # loss += (l1_loss*0.1)
                # loss += (l1_loss*0.07)
                # loss += (l1_loss*0.01)

                loss = loss + l1_loss
                # loss = loss + 0.7*l1_loss
                # loss = loss + 1.25*l1_loss

            loss.backward()

            optimizer.step()

            # #Cal acc
            # acc = train_acc(outputs, targets)
            # print(f"Accuracy on batch {idx}: {acc}")

            inputs_list.append(inputs.detach().cpu().numpy())
            targets_list.append(targets.detach().cpu().numpy())
            preds_list.append(outputs.detach().cpu().numpy())
        

        # # metric on all batches using custom accumulation
        # acc = train_acc.compute()
        # print(f"Accuracy on all data: {acc}")

       
    # metric on all batches using custom accumulation
    # acc = train_acc.compute()
    inputs_list = np.vstack(inputs_list)
    targets_list = np.hstack(targets_list)
    preds_list = np.vstack(preds_list)
    # acc = (preds_list == targets_list).mean()
    # acc = (preds_list.argmax(axis = 1) == targets_list).mean()
    if isinstance(criterion, nn.BCEWithLogitsLoss):
        acc = ( (torch.sigmoid(torch.tensor(preds_list)) > 0.5).numpy().squeeze(1) == targets_list).mean()
    else:
        acc = (torch.softmax(torch.tensor(preds_list), dim = 1).argmax(1).numpy() == targets_list).mean()
    # acc = 1 - np.linalg.norm(preds_list - targets_list)

    print(f"Train accuracy on all data: {acc}")

    return model, acc


def test(model, test_dataloader, criterion, previous_model = False,
            return_features = False, return_prob_preds = False, device = "cuda:1"):


    model.eval()


    with torch.no_grad():

        
        targets_list = []
        preds_list = []
        features_list = []
        for idx, (filenames, inputs, targets) in enumerate(tqdm.tqdm(test_dataloader)):

            inputs = inputs.to(device)
            targets = targets.to(device)

            if previous_model:
                outputs = model(inputs, previous = True)
            else:
                if return_features:
                    outputs, features = model(inputs, return_features = True)
                    features_list.append(features.detach().cpu().numpy())
                    # features_list.append(torch.cat((inputs, features), dim = 1).detach().cpu().numpy())
                else:
                    outputs = model(inputs)

            loss = computeLoss(criterion, outputs, targets)

            targets_list.append(targets.detach().cpu().numpy())
            preds_list.append(outputs.detach().cpu().numpy())


    targets_list = np.hstack(targets_list)
    preds_list = np.vstack(preds_list)
    # acc = (preds_list == targets_list).mean()
    # acc = (preds_list.argmax(axis = 1) == targets_list).mean()
    if isinstance(criterion, nn.BCEWithLogitsLoss):
        acc = ( (torch.sigmoid(torch.tensor(preds_list)) > 0.5).numpy().squeeze(1) == targets_list).mean()
    else:
        acc = (torch.softmax(torch.tensor(preds_list), dim = 1).argmax(1).numpy() == targets_list).mean()
    # acc = 1 - np.linalg.norm(preds_list - targets_list)

    print(f"Test accuracy on all data: {acc}")

   

    model.train()



    if return_features:
        features_list = np.vstack(features_list)
        
        if return_prob_preds:
            # return features_list, preds_list, model_results_dict['accuracy']
            return features_list, preds_list, acc
        
        # return features_list, model_results_dict['accuracy']
        return features_list, targets_list, acc

    if return_prob_preds:
        # return preds_list, model_results_dict['accuracy']
        return preds_list, acc

    # return model_results_dict['accuracy']
    return acc


    

def computeLoss(criterion, outputs, targets):

    if isinstance(criterion, nn.CrossEntropyLoss):
        if len(targets.shape) > 1:
            targets = targets.squeeze(1)

    if isinstance(criterion, nn.BCEWithLogitsLoss):
        if len(outputs.shape) > 1:
            outputs = outputs.squeeze(1)
        loss = criterion(outputs, targets.float())
    else:
        loss = criterion(outputs, targets)

    return loss




def plotPreds(inputs_list, targets_list, preds_list,
        fig_name = "train_pred_curves.png"):

    showFig = True
    path  = "/home/grg/Research/DARPA-Pneumothorax/results/DeepSymbolicRegression_Exps/"


    if inputs_list.shape[1] == 1:

        plt.figure()

        # plt.plot(inputs_list, targets_list, label = "target")
        # plt.plot(inputs_list, preds_list, label = "pred")
        plt.scatter(inputs_list, targets_list, label = "target")
        plt.scatter(inputs_list, preds_list, label = "pred")

        plt.xlabel("x")
        plt.ylabel("equation")
        plt.title("Traget and Predicted equation")
        plt.legend()
        # plt.grid(True)

    # elif inputs_list.shape[1] == 2:
    elif inputs_list.shape[1] >= 2:

        # plt.subplot(2, 1, 1)
        fig, ax =  plt.subplots(1,2)
        ax[0].scatter(inputs_list[:, 0], targets_list, label = "target")
        ax[0].scatter(inputs_list[:, 0], preds_list, label = "pred")

        ax[0].set_xlabel("x")
        ax[0].set_ylabel("equation")
        ax[0].set_title("Traget and Predicted equation")
        ax[0].legend()

        # plt.subplot(2, 1, 2)
        ax[1].scatter(inputs_list[:, 1], targets_list, label = "target")
        ax[1].scatter(inputs_list[:, 1], preds_list, label = "pred")

        ax[1].set_xlabel("y")
        ax[1].set_ylabel("equation")
        ax[1].set_title("Traget and Predicted equation")
        ax[1].legend()


    plt.savefig(os.path.join(path, fig_name))
    
    if showFig:
        plt.show()

    # plt.close()

    plot3D = True
    # if plot3D and inputs_list.shape[1] == 2:      
    if plot3D and inputs_list.shape[1] >= 2:       
        # Creating figure
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
        
        # Creating plot
        ax.scatter3D(inputs_list[:, 0], inputs_list[:, 1], targets_list, color = "green")
        ax.scatter3D(inputs_list[:, 0], inputs_list[:, 1], preds_list, color = "orange")
        plt.title(fig_name)
        
        # show plot
        if showFig:
            plt.show()

    # plt.close()

