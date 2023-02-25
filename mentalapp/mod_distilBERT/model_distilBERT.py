"""File defining the neural network model class based on DistilBERT"""

from torch import nn
from transformers import DistilBertModel


class DistilBERTSentimentClassifier(nn.Module):
    """
    Class implementing a DistilBERT classification model with all its layers.
    """

    def __init__(self, number_classes, pre_trained_model_name, drop_out):
        """
        Init function of DistilBERTSentimentClassifier class
        :param number_classes: Number of final classification types an input text can have
        :type: Int
        :param pre_trained_model_name: Name of the selected DistilBERT pre-trained model
        :type: String
        :param drop_out: Number per one of neurons that are deactivated in the network
        :type: Decimal
        :return: Nothing
        """

        #Initializer required for the model
        super(DistilBERTSentimentClassifier,self).__init__()

        #DistilBERT neural network model
        self.distil_bert = DistilBertModel.from_pretrained(pre_trained_model_name)
        #Extra specific layer of neurons to avoid overfitting
        self.drop = nn.Dropout(p = drop_out)
        #Extra neuron layer for text classification
        #It has as many input neurons as DistilBERT network has output neurons
        #Number of output neurons is equal to number of possible classifications
        self.linear = nn.Linear(self.distil_bert.config.hidden_size,number_classes)


    def forward(self, input_ids, attention_mask):
        """
        Function needed in Pytorch to specify order of layers
        :param input_ids: Representative identifiers of input data
        :type: Tensor
        :param attention_mask: Attention mask for transformers technology
        :type: Tensor
        :return: Output of complete model
        :type: Tensor
        """

        #Encoded input and the encoding of the classification token resulting from passing through DistilBERT layer are obtained
        distil_bert_output = self.distil_bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            return_dict = True
        )

        #Encoding of the classification token is passed as input data of the drop layer
        #This vector contains all the essence of input data
        drop_output = self.drop(distil_bert_output[0])

        #Output vector of drop layer is passed as input data to linear layer and final classification is obtained
        output = self.linear(drop_output)

        #Final classification calculated by passing through all layers of model is returned
        return output