"""Main application file. Contains the general operating logic"""

import __init__ as init
import numpy
import torch
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.prompt import Prompt

from TUI_menu import option_menu, welcome_menu, help_menu
from mod_dataset.dataset import Dataset
from mod_BERT.model_BERT import BERTSentimentClassifier
from trainer import train_model, eval_model
from menu_options import *

#Load constants and predefined application parameters
configuration_main = init.load_config_mentalapp()


def dataset_initialize():
    """
    Initialisation of the default dataset
    :return: Complete dataset, Training dataset, Evaluation dataset with correct format
    :type: Dataset
    """

    #Dataset with all data
    complete_dataset = Dataset()
    #Dataset with data for training
    train_dataset = Dataset()
    #Dataset with data for evaluation
    test_dataset = Dataset()

    #All data are read from the source file
    raw_data = complete_dataset.read_file(configuration_main['FILE_DATASET_NAME'])
    #A sample of the data is taken for testing.
    raw_data = raw_data[0:10]
    #The data is divided between training and evaluation. The parameter test_size marks the percent per one of data for evaluation.
    train_raw_data,test_raw_data = train_test_split(raw_data, test_size = 0.2, random_state = configuration_main['RANDOM_SEED'])
    
    #Dataframes are put in the required format.
    complete_dataset.format_dataset(raw_data)
    train_dataset.format_dataset(train_raw_data)
    test_dataset.format_dataset(test_raw_data)

    return complete_dataset, train_dataset, test_dataset


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


def training(model, device, train_data_loader, number_train_data, test_data_loader, number_test_data):
    """
    The pre-training module and the additional layers added are trained and evaluated.
    :param model: Neural network model
    :type: MODELSentimentClassifier
    :param device: Calculation optimizer
    :type: Torch Device
    :param train_data_loader: Dataset custom Pytorch for training
    :type: DataLoader
    :param number_train_data: Total number of training data
    :type: Int
    :param test_data_loader: Dataset customized Pytorch for evaluation
    :type: DataLoader
    :param number_test_data: Total number of evaluating data
    :type: Int
    :return: Nothing
    """

    #Optimizer is created and a learning rate lr is assigned.
    optimizer = AdamW(model.parameters(), lr=2e-5)

    #Total number of training iterations
    total_steps = len(train_data_loader) * configuration_main['EPOCHS']

    #Function to reduce the learning rate
    scheduler = get_linear_schedule_with_warmup(
        optimizer, #Optimizer function
        num_warmup_steps = 0, #Number of iterations the model waits for to start reducing the learning rate
        num_training_steps = total_steps #Total number of training steps
    )

    #Error function to be minimized
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    #For each epoch, the model is trained and validated.
    for epoch in range(configuration_main['EPOCHS']):
        print('Epoch {} de {}'.format(epoch + 1, configuration_main['EPOCHS']))
        print('--------------------')

        #Model training and parameter update
        model, optimizer, scheduler, train_accuracy, train_loss = train_model(
            model, train_data_loader, loss_fn, optimizer, device, scheduler, number_train_data
        )

        #Model validated and parameter update
        test_accuracy, test_loss = eval_model(
            model, test_data_loader, loss_fn, device, number_test_data
        )

        print('Entrenamiento: Loss:{}, Accuracy:{}'.format(train_loss, train_accuracy))
        print('Validación: Loss:{}, Accuracy:{}'.format(test_loss, test_accuracy))
        print('')


def main():
    """Main function of the application. Manages the menu and calls to the main blocks
    :return: Nothing
    """

    welcome_menu()
    
    #Random initialization of the weights and parameters of Pytorch model
    numpy.random.seed(configuration_main['RANDOM_SEED'])
    torch.manual_seed(configuration_main['RANDOM_SEED'])

    #Use of the computational optimizer if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #Raw dataset initialization
    complete_dataset,train_dataset,test_dataset = dataset_initialize()

    #Options available to the user
    selected_option = 0
    menu_options = {
        '1': "use_classify_model(configuration_main, device)",
        '2': "training_model_scratch(configuration_main, device, train_dataset)",
        '3': "evaluating_model_pretraining(configuration_main, device, test_dataset)",
        '4': "pass",
        '5': "pass",
        '6': "help_menu()",
    }
    
    #As long as user does not select the exit option, program continues to run
    while selected_option != "7":
        option_menu()

        custom_theme = Theme({"success": "green", "error": "red"})
        console = Console(theme = custom_theme)
        selected_option = Prompt.ask("Seleccione una opción")
        console.print(Panel.fit("Opción: " + selected_option))

        #User option is executed if possible
        if selected_option in menu_options.keys():
            eval(menu_options[selected_option])
        else:
            if selected_option == '7':
                pass
            else:
                console.print("[error] Opción incorrecta [/error], por favor selecione una de las [success] opciones disponibles [/success]")
                print("")

    #Function transforming input data into special codes (tokens) for BERT model
    tokenizer = BertTokenizer.from_pretrained(configuration_main['PRE_TRAINED_MODEL_NAME']['Bert'])

    #Creation of Pytorch dataset for training and evaluation
    train_data_loader = data_loader(train_dataset,tokenizer,configuration_main['MAX_DATA_LEN'],configuration_main['BATCH_SIZE'],configuration_main['DATALOADER_NUM_WORKERS'])
    test_data_loader = data_loader(test_dataset,tokenizer,configuration_main['MAX_DATA_LEN'],configuration_main['BATCH_SIZE'],configuration_main['DATALOADER_NUM_WORKERS'])

    #Creation of BERT model
    model = BERTSentimentClassifier(configuration_main['NUM_TYPES_CLASSIFICATION_CLASSES'], configuration_main['PRE_TRAINED_MODEL_NAME']['Bert'], configuration_main['DROP_OUT_BERT'])

    #Model is taken to the GPU if available
    model = model.to(device)

    #Model training and evaluation
    training(model, device, train_data_loader, len(train_dataset), test_data_loader, len(test_dataset))
    

if __name__ == "__main__":
    main()
