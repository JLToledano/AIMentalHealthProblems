"""File defining the trainer and evaluator of a neural network model"""

import torch

from torch import nn
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from rich.console import Console
from rich.align import Align
from rich.panel import Panel
from rich.theme import Theme
from rich.table import Table
from rich.table import Column


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

    return model, optimizer, scheduler


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


def metrics_model(labels, predictions):
    """
    Calculates the metrics necessary for testing the performance and evolution of neural model
    :param labels: Original data classification labels
    :type: Tensor
    :param predictions: Data classification labels predicted by the model
    :type: Tensor
    :return: Nothing
    """

    #Customization of the console with special predefined styles
    custom_theme = Theme({"parameter":"purple"})
    console = Console(theme = custom_theme)

    #Section header design
    section_title_message = """RESULTADOS MEDIDAS"""
    section_title_message_align = Align(section_title_message, align="center")
    console.print(Panel(section_title_message_align, style="bold"))

    #Confusion Matrix
    confusion = confusion_matrix(labels, predictions)
    true_negative_panel = Panel(Align("[parameter]TN[/parameter] " + str(confusion[0][0]), align="center"), title="True Negative")
    false_positive_panel = Panel(Align("[parameter]FP[/parameter] " + str(confusion[0][1]), align="center"), title="False Positive")
    false_negative_panel = Panel(Align("[parameter]FN[/parameter] " + str(confusion[1][0]), align="center"), title="False Negative")
    true_positive_panel = Panel(Align("[parameter]TP[/parameter] " + str(confusion[1][1]), align="center"), title="True Positive")

    #Accuracy
    #Return the fraction of correctly classified samples (float)
    accurancy = accuracy_score(labels, predictions)
    accurancy_panel = Panel(Align(str(accurancy), align="center"), title="Accurancy")

    #Recall
    #The recall is the ratio tp / (tp + fn)
    #The recall is intuitively the ability of the classifier to find all the positive samples
    recall = recall_score(labels, predictions, average="binary", zero_division = 0)
    recall_panel = Panel(Align(str(recall), align="center"), title="Recall")

    #Precision
    #The precision is the ratio tp / (tp + fp)
    #The precision is intuitively the ability of the classifier not to label as positive a sample that is negative
    precision = precision_score(labels, predictions, average="binary", zero_division = 0)
    precision_panel = Panel(Align(str(precision), align="center"), title="Precision")

    #F1
    #The F1 score can be interpreted as a harmonic mean of the precision and recall, 
    #where an F1 score reaches its best value at 1 and worst score at 0
    f1 = f1_score(labels, predictions, average="binary", zero_division = 0)
    f1_panel = Panel(Align(str(f1), align="center"), title="F1")
    
    #General table design
    measurements_table = Table(
        Column(header="Matriz de Confusión", justify="center"),
        Column(header="Otras Medidas", justify="center"),
        expand=True
    )

    #Design of confusion matrix table (left side general table)
    confusion_matrix_table = Table.grid(expand=True)
    confusion_matrix_table.add_column(justify="center")
    confusion_matrix_table.add_column(justify="center")
    confusion_matrix_table.add_row(true_negative_panel,false_positive_panel)
    confusion_matrix_table.add_row(false_negative_panel,true_positive_panel)

    #Design of table of other measurements (right side of general table)
    other_measurements_table = Table.grid(expand=True)
    other_measurements_table.add_column(justify="center")
    other_measurements_table.add_column(justify="center")
    other_measurements_table.add_row(accurancy_panel,recall_panel)
    other_measurements_table.add_row(precision_panel,f1_panel)

    #Connection of subtables with the general table
    measurements_table.add_row(confusion_matrix_table,other_measurements_table)

    #The set of tables is printed
    console.print(measurements_table)
    console.print('\n')