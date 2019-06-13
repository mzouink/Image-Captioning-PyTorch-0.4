import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size,  n_layers=1):
        super().__init__()
        
        ## TODO: define the LSTM
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, batch_first=True)
        
        ## TODO: define the final, fully-connected output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.hidden = (torch.zeros(1,1,hidden_size),torch.zeros(1,1,hidden_size))
    
    def forward(self, features, captions):
        captions = self.embed(captions[:,:-1])
        
        features = features.unsqueeze(1)
        
        inputs = torch.cat((features,captions),1)
        
        lstm_output,_ = self.lstm(inputs,None)
        
        outputs = self.fc(lstm_output)

        return  outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        result = []
        for i in range(max_len):
            outputs, hidden = self.lstm(inputs, states)
            outputs = self.fc(outputs.squeeze(1))
            target_index = outputs.max(1)[1]
            result.append(target_index.item())
            inputs = self.embed(target_index).unsqueeze(1)
        return result