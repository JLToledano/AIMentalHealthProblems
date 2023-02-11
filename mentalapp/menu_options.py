import os
import pathlib
import numpy
import torch
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from rich.prompt import Prompt, IntPrompt, FloatPrompt
from rich.prompt import Confirm

from mod_BERT.model_BERT import BERTSentimentClassifier
from trainer import train_model, eval_model

def save_model(model):
    """
    Configuration of a trained model is stored in a file for possible future use
    :param model: Trained model
    :type: MODELSentimentClassifier
    :return: Nothing
    """

    name_with_blank_spaces = True

    #If file name specified by user contains blanks, another name is requested
    while name_with_blank_spaces:
        model_name = input("Escoja un nombre para el modelo: ")

        if " " in model_name:
            name_with_blank_spaces = True
            print("No se recomienda el uso de espacios en un nombre de fichero")
        else:
             name_with_blank_spaces = False

    #The file path is set
    model_path = "models\{}.pt".format(model_name)

    #Torch model is stored in selected path
    torch.save(model,os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path))


def load_model():
    """
    A model pre-trained by the user is loaded
    :return: Model
    :type: MODELSentimentClassifier
    """

    list_pretraining_models = []
    number_file = 0
    selected_file = False

    #The directory where models are located is established
    path_models = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    address = pathlib.Path(path_models)

    print("Ficheros disponibles:")

    #Each file available in directory is printed and added to list
    for file in address.iterdir():
        number_file += 1
        list_pretraining_models.append(file.name)
        print("    " + str(number_file) + ". " + file.name)

    #User is asked to choose a pre-trained model and is not stopped until a valid one is chosen
    while not selected_file:
        try:
            number_file = int(input("Seleccione el número del modelo que desea: "))
            name_file = list_pretraining_models[number_file - 1]
            #Torch model is loaded
            model = torch.load(os.path.join(path_models, name_file))
            selected_file = True
        except:
            print("Número de fichero no válido")

    return model


def data_loader(dataset,tokenizer,max_len,batch_size,num_workers):
    """
    Adding the necessary structure to the dataset to adapt it to Pytorch
    :param dataset: Generic dataset with data
    :type: Dataset
    :param tokenizer: Function that transforms input data into special codes (tokens)
    :type: Tokenizer
    :param max_len: Maximum number of words accepted by model as input parameter
    :type: Int
    :param batch_size: Lot size. Number of data to be inserted into neural network at a time
    :type: Int
    :param num_workers: Number of processes running in parallel. Analyzes x data in parallel
    :type: Int
    :return: Custom Pytorch dataset
    :type: DataLoader
    """

    dataset.set_tokenizer(tokenizer)
    dataset.set_max_len(max_len)

    #Pytorch-specific DataLoader is created
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def training_model_scratch(configuration_main, device, train_dataset):
    """
    The pre-training model and the additional layers added are trained
    :param configuration_main: Training configurations
    :type: dict[String:String]
    :param device: Calculation optimizer
    :type: Torch Device
    :param train_dataset: Dataset with data for training
    :type: Dataset
    :return: Nothing
    """
    #Function transforming input data into special codes (tokens) for BERT model
    tokenizer = BertTokenizer.from_pretrained(configuration_main['PRE_TRAINED_MODEL_NAME']['Bert'])

    #Creation of Pytorch dataset for training
    train_data_loader = data_loader(train_dataset,tokenizer,configuration_main['MAX_DATA_LEN'],configuration_main['BATCH_SIZE'],configuration_main['DATALOADER_NUM_WORKERS'])

    #Creation of BERT model
    model = BERTSentimentClassifier(configuration_main['NUM_TYPES_CLASSIFICATION_CLASSES'], configuration_main['PRE_TRAINED_MODEL_NAME']['Bert'], configuration_main['DROP_OUT_BERT'])

    #Model is taken to the GPU if available
    model = model.to(device)

    #Optimizer is created and a learning rate lr is assigned.
    optimizer = AdamW(model.parameters(), lr=2e-5)

    #Total number of training iterations
    total_steps = len(train_data_loader) * configuration_main['EPOCHS']

    #Total number of training data
    number_train_data = len(train_dataset)

    #Function to reduce the learning rate
    scheduler = get_linear_schedule_with_warmup(
        optimizer, #Optimizer function
        num_warmup_steps = 0, #Number of iterations the model waits for to start reducing the learning rate
        num_training_steps = total_steps #Total number of training steps
    )

    #Error function to be minimized
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    #For each epoch, the model is trained.
    for epoch in range(configuration_main['EPOCHS']):
        print('Epoch {} de {}'.format(epoch + 1, configuration_main['EPOCHS']))
        print('--------------------')

        #Model training and parameter update
        model, optimizer, scheduler, train_accuracy, train_loss = train_model(
            model, train_data_loader, loss_fn, optimizer, device, scheduler, number_train_data
        )

        print('Entrenamiento: Loss:{}, Accuracy:{}'.format(train_loss, train_accuracy))
        print('')
    
    #Trained model is stored
    save_model(model)


def evaluating_model_pretraining(configuration_main, device, test_dataset):
    """
    The pre-training model and the additional layers added are evaluated
    :param configuration_main: Evaluating configurations
    :type: dict[String:String]
    :param device: Calculation optimizer
    :type: Torch Device
    :param test_dataset: Dataset with data for evaluating
    :type: Dataset
    :return: Nothing
    """

    #Pre-trained Torch model is loaded
    model = load_model()

    #Function transforming input data into special codes (tokens) for BERT model
    tokenizer = BertTokenizer.from_pretrained(configuration_main['PRE_TRAINED_MODEL_NAME']['Bert'])

    #Creation of Pytorch dataset for evaluating
    test_data_loader = data_loader(test_dataset,tokenizer,configuration_main['MAX_DATA_LEN'],configuration_main['BATCH_SIZE'],configuration_main['DATALOADER_NUM_WORKERS'])

    #Total number of evaluating data
    number_test_data = len(test_dataset)

    #Error function to be minimized
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    #For each epoch, the model is validated.
    for epoch in range(configuration_main['EPOCHS']):
        print('Epoch {} de {}'.format(epoch + 1, configuration_main['EPOCHS']))
        print('--------------------')

        #Model validated and parameter update
        test_accuracy, test_loss = eval_model(
            model, test_data_loader, loss_fn, device, number_test_data
        )

        print('Validación: Loss:{}, Accuracy:{}'.format(test_loss, test_accuracy))
        print('')


def use_classify_model(configuration_main, device):
    """
    Use pre-training model with a user message
    :param configuration_main: Evaluating configurations
    :type: dict[String:String]
    :param device: Calculation optimizer
    :type: Torch Device
    :return: Nothing
    """

    #Pre-trained Torch model is loaded
    model = load_model()

    #Function transforming input data into special codes (tokens) for BERT model
    tokenizer = BertTokenizer.from_pretrained(configuration_main['PRE_TRAINED_MODEL_NAME']['Bert'])

    #User mesagge
    text = input("Inserte el texto que quiere clasificar:\n")

    #Coding of input data
    encoding_text = tokenizer.encode_plus(
        text, #Original message
        max_length = configuration_main['MAX_DATA_LEN'], #Maximum number of tokens (counting special tokens)
        truncation = True, #Ignoring tokens beyond the set number of tokens
        add_special_tokens = True, #Special tokens [CLS], [SEP] and [PAD] added
        return_token_type_ids = False,
        padding = 'max_length', #If total number of tokens is less than the established maximum, it is filled with [PAD] until the maximum is reached
        return_attention_mask = True, #Model is instructed to pay attention only to non-empty tokens during training
        return_tensors = 'pt' #Final result of the encoder in Pytorch numbers
    )

    input_ids = encoding_text['input_ids'].to(device) #Numeric input tokens and special tokens
    attention_mask = encoding_text['attention_mask'].to(device) #Attention mask

    #Model outputs are computed
    outputs = model(input_ids = input_ids, attention_mask = attention_mask)
    #Predictions are calculated. Maximum of 2 outputs is taken
    #If first one is the maximum, suicide, if second one is the maximum, non-suicide
    _, preds = torch.max(outputs, dim = 1)

    if preds:
        print("Clasificación: Suicide")
    else:
        print("Clasificación: Non-Suicide")


def customize_parameter_configuration(configuration_main):
    """
    Function for changing value of the general training and evaluation settings of neural network
    :param configuration_main: Dictionary with general settings
    :type: dict[String:String]
    :return: General settings modified or same if user decides not to make changes
    :type: dict[String:String]
    """
    
    #User is prompted for maximum data length by displaying the default option
    configuration_main['MAX_DATA_LEN'] = IntPrompt.ask("Por favor indique el número máximo de palabras que admita el modelo", default=configuration_main['MAX_DATA_LEN'])

    #User is prompted for batch size by displaying the default option
    configuration_main['BATCH_SIZE'] = IntPrompt.ask("Por favor indique el tamaño de lote", default=configuration_main['BATCH_SIZE'])

    #User is prompted for number of epochs by displaying the default option
    configuration_main['EPOCHS'] = IntPrompt.ask("Por favor indique el número de épocas (iteraciones)", default=configuration_main['EPOCHS'])

    #User is prompted for number of parallel processes by displaying the default option
    configuration_main['DATALOADER_NUM_WORKERS'] = IntPrompt.ask("Por favor indique el número de procesos paralelos", default=configuration_main['DATALOADER_NUM_WORKERS'])

    #User is prompted for drop out per one by displaying the default option
    configuration_main['DROP_OUT_BERT'] = FloatPrompt.ask("Por favor indique el tanto por uno de drop out para BERT", default=configuration_main['DROP_OUT_BERT'])

    confirm_changes = Confirm.ask("¿Estás seguro de los cambios realizados?")

    #If user confirms changes are perpetuated
    if not confirm_changes:
        #User is given possibility to execute changes again.
        configuration_main = customize_parameter_configuration(configuration_main)
        
    return configuration_main


def assign_new_dataset():
    """
    Function for assigning a new dataset file as the source of the training and evaluation data
    :return: Name of new file
    :type: String
    """

    #User is prompted for a filename by displaying the default option
    file_name = Prompt.ask("Por favor indique el nombre de su fichero", default="Suicide_Detection.csv")

    return file_name