"""File defining the trainer and evaluator of a neural network model"""

import numpy as np
import torch
from torch import nn
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, number_data):
    """
    Function that trains a complete model with input data and configurations provided
    :param model: Complete neural model
    :type: MODELSentimentClassifier
    :param data_loader: Function that loads training data
    :type: DataLoader
    :param loss_fn: Error function
    :type: CrossEntropyLoss
    :param optimizer: Optimization function
    :type: AdamW
    :param device: GPU used if available
    :type: Device
    :param scheduler: Function to progressively reduce learning rate
    :type: LambdaLR
    :param number_data: Number of training data
    :type: Int
    :return: Trained model
    :type: MODELSentimentClassifier
    :return: Upgraded optimizer
    :type: AdamW
    :return: Upgraded function to progressively reduce learning rate
    :type: LambdaLR
    :return: Training accuracy
    :type: Tensor
    :return: Mean error value
    :type: Float64
    """

    #The model is put into training mode
    model = model.train()
    #To store value of error in each iteration
    losses = []
    #Set initial training accuracy to 0
    correct_predictions = 0

    for batch in data_loader:
        #Inputs_ids are extracted from batch data and sent to GPU to speed up training
        input_ids = batch['input_ids'].to(device)
        #Attention mask is extracted from batch data and sent to GPU to speed up training
        attention_mask = batch['attention_mask'].to(device)
        #Labels are extracted from batch data and sent to GPU to speed up training
        labels = batch['text_clasification'].to(device)
        #Model outputs are computed
        outputs = model(input_ids = input_ids, attention_mask = attention_mask)
        #Predictions are calculated (in this case or performed by BERT).Maximum of 2 outputs is taken
        #If first one is the maximum, suicide, if second one is the maximum, non-suicide
        _, preds = torch.max(outputs, dim = 1)

        #Error calculation
        loss = loss_fn(outputs, labels)
        #Calculation of successes. Accumulated sum of the predictions equals original labels
        correct_predictions += torch.sum(preds == labels)
        #Error made is added to error list
        losses.append(loss.item())

        #Error is back-propagated
        loss.backward()
        #Gradient is prevented from increasing too much so as not to slow down progress of training with excessively large jumps
        #Gradient value is always kept between -1 and 1.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

        #Optimizer (weights) is updated.
        optimizer.step()
        #Training rate is updated
        scheduler.step()
        #Gradients are reset for next iteration
        optimizer.zero_grad()

        #Calculate the metrics required for the design study
        metrics_model(labels, preds)

    return model, optimizer, scheduler, correct_predictions.double() / number_data, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, number_data):
    """
    Function that evaluating a complete model with input data and configurations provided
    :param model: Complete neural model
    :type: MODELSentimentClassifier
    :param data_loader: Function that loads training data
    :type: DataLoader
    :param loss_fn: Error function
    :type: CrossEntropyLoss
    :param device: GPU used if available
    :type: Device
    :param number_data: Number of training data
    :type: Int
    :return: Training accuracy
    :type: Tensor
    :return: Mean error value
    :type: Float64
    """

    #The model is put into evaluating mode
    model = model.eval()
    #To store value of error in each iteration
    losses = []
    #Set initial evaluating accuracy to 0
    correct_predictions = 0

    #It is indicated that no model parameters should be modified
    with torch.no_grad():
        for batch in data_loader:
            #Inputs_ids are extracted from batch data and sent to GPU to speed up evaluation
            input_ids = batch['input_ids'].to(device)
            #Attention mask is extracted from batch data and sent to GPU to speed up evaluation
            attention_mask = batch['attention_mask'].to(device)
            #Labels are extracted from batch data and sent to GPU to speed up evaluation
            labels = batch['text_clasification'].to(device)
            #Model outputs are computed
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)
            #Predictions are calculated (in this case or performed by BERT).Maximum of 2 outputs is taken
            #If first one is the maximum, suicide, if second one is the maximum, non-suicide.
            _, preds = torch.max(outputs, dim = 1)

            #Error calculation
            loss = loss_fn(outputs, labels)
            #Calculation of successes. Accumulated sum of the predictions equals original labels
            correct_predictions += torch.sum(preds == labels)
            #Error made is added to error list
            losses.append(loss.item())

            #Calculate the metrics required for the design study
            metrics_model(labels, preds)

    return correct_predictions.double() / number_data, np.mean(losses)


def metrics_model(labels, predictions):
    """
    Calculates the metrics necessary for testing the performance and evolution of neural model
    :param labels: Original data classification labels
    :type: Tensor
    :param predictions: Data classification labels predicted by the model
    :type: Tensor
    :return: Nothing
    """

    print('MEDIDAS')
    print('-----------------------')
    #Confusion Matrix
    confusion = confusion_matrix(labels, predictions)
    print(confusion)

    #Accuracy
    accurancy = accuracy_score(labels, predictions)
    print(accurancy)

    #Recall
    recall = recall_score(labels, predictions, average=None, zero_division = 0)
    print(recall)

    #Precision
    precision = precision_score(labels, predictions, average=None, zero_division = 0)
    print(precision)

    #F1
    f1 = f1_score(labels, predictions, average=None, zero_division = 0)
    print(f1)

    print('-----------------------')