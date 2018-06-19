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
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=3):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

        
    
    def forward(self, features, captions):
        #print("FFFFFFFFFFFFFFFFFFFFFFFFFFFF", features.shape)
        #print("CAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", captions.shape)
        #embeddings = self.embed(captions)
        embeddings = self.embed(captions[:, :-1])
        #print("Emmmmmmmmmmmmmmmmmmmmm", embeddings.shape)
        #inputs = torch.cat(features, embeddings).view(len(features), 1, -1)
        #features= features.view(features.shape[0],-1, features.shape[1])
        #print('fffffffffffffffffffffffffffffffffffffffff', features.shape)

        #inputs = torch.cat([features, embeddings], 0)
        
        #inputs = torch.cat((features, embeddings), 1)
        
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)

        #print("innnnnnnnnnnnnnnnnnnnnn", inputs.shape)
        #inputs = inputs.squeeze(1)
        #print("aftttterrrrrrrrrrrrrrr sqqqqqqqqqqqq", inputs.shape)

        #hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
        out, hidden = self.lstm(inputs)
        #print("lssss outttttttttttttttttt", out.shape)
        outputs = self.linear(out)
        #print("OOOOOOOOOOOOOOOOOOOOOOOOOO", outputs.shape)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        preds = []
        for i in range(max_len):
            out, states = self.lstm(inputs, states)          
            outputs = self.linear(out.squeeze(1))            
            predicted = outputs.max(1)[1]                       
            preds.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)
        return list(preds)
