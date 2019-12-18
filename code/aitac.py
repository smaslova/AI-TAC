import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import random
import numpy as np
import copy

# Convolutional neural network
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        
        # for layer one, separate convolution and relu step from maxpool and batch normalization
        # to extract convolutional filters
        self.layer1_conv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=300,
                      kernel_size=(4, 19),
                      stride=1,
                      padding=0),  # padding is done in forward method along 1 dimension only
            nn.ReLU())

        self.layer1_process = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,3), padding=(0,1)),
            nn.BatchNorm2d(300))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=300,
                      out_channels=200,
                      kernel_size=(1, 11),
                      stride=1,
                      padding=0),  # padding is done in forward method along 1 dimension only
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4), padding=(0,1)),
            nn.BatchNorm2d(200))

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=200,
                      out_channels=200,
                      kernel_size=(1, 7),
                      stride=1,
                      padding=0),  # padding is done in forward method along 1 dimension only
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1,4), padding=(0,1)),
            nn.BatchNorm2d(200))

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=1000,
                      out_features=1000),
            nn.ReLU(),
            nn.Dropout(p=0.03))

        self.layer5 = nn.Sequential(
            nn.Linear(in_features=1000,
                      out_features=1000),
            nn.ReLU(),
            nn.Dropout(p=0.03))

        self.layer6 = nn.Sequential(
                nn.Linear(in_features=1000,
                          out_features=num_classes))#,
                #nn.Sigmoid())


    def forward(self, input):
        # run all layers on input data
        # add dummy dimension to input (for num channels=1)
        input = torch.unsqueeze(input, 1)

        # Run convolutional layers
        input = F.pad(input, (9, 9), mode='constant', value=0) # padding - last dimension goes first
        out = self.layer1_conv(input)
        activations = torch.squeeze(out)
        out = self.layer1_process(out)
        
        out = F.pad(out, (5, 5), mode='constant', value=0)
        out = self.layer2(out)

        out = F.pad(out, (3, 3), mode='constant', value=0)
        out = self.layer3(out)
        
        # Flatten output of convolutional layers
        out = out.view(out.size()[0], -1)
        
        # run fully connected layers
        out = self.layer4(out)
        out = self.layer5(out)
        predictions = self.layer6(out)
        
        activations, act_index = torch.max(activations, dim=2)
        
        return predictions, activations, act_index
      
# define model for extracting motifs from first convolutional layer
# and determining importance of each filter on prediction
class motifCNN(nn.Module):
            def __init__(self, original_model):
                super(motifCNN, self).__init__()
                self.layer1_conv = nn.Sequential(*list(original_model.children())[0])
                self.layer1_process = nn.Sequential(*list(original_model.children())[1])
                self.layer2 = nn.Sequential(*list(original_model.children())[2])
                self.layer3 = nn.Sequential(*list(original_model.children())[3])
                
                self.layer4 = nn.Sequential(*list(original_model.children())[4])
                self.layer5 = nn.Sequential(*list(original_model.children())[5])
                self.layer6 = nn.Sequential(*list(original_model.children())[6])
                

            def forward(self, input):
                # add dummy dimension to input (for num channels=1)
                input = torch.unsqueeze(input, 1)
                
                # Run convolutional layers
                input = F.pad(input, (9, 9), mode='constant', value=0) # padding - last dimension goes first
                out= self.layer1_conv(input)
                layer1_activations = torch.squeeze(out)
                
                #do maxpooling and batch normalization for layer 1
                layer1_out = self.layer1_process(out)
                layer1_out = F.pad(layer1_out, (5, 5), mode='constant', value=0)
                
                #calculate average activation by filter for the whole batch
                filter_means_batch = layer1_activations.mean(0).mean(1)
            
                # run all other layers with 1 filter left out at a time
                batch_size = layer1_out.shape[0]
                predictions = torch.zeros(batch_size, 300,  81)

                #filter_matches = np.load("../outputs/motifs2/run2_motif_matches.npy")

                for i in range(300):
                    #modify filter i of first layer output
                    filter_input = layer1_out.clone()

                    filter_input[:,i,:,:] = filter_input.new_full((batch_size, 1, 94), fill_value=filter_means_batch[i])
                    #filter_input[:,i,:,:] = filter_input.new_full((batch_size, 1, 94), fill_value=0)

                    #match = filter_matches[filter_matches[:,0]==i,][:,1]
                    #for j in match:
                    #    filter_input[:,j,:,:] = filter_input.new_full((batch_size, 1, 94), fill_value=filter_means_batch[j])
                    
                    out = self.layer2(filter_input)
                    out = F.pad(out, (3, 3), mode='constant', value=0)
                    out = self.layer3(out)
                    
                    # Flatten output of convolutional layers
                    out = out.view(out.size()[0], -1)
                    # run fully connected layers
                    out = self.layer4(out)
                    out = self.layer5(out)
                    out = self.layer6(out)
                    
                    predictions[:,i,:] = out

                    activations, act_index = torch.max(layer1_activations, dim=2)

                return predictions, layer1_activations, act_index
           
    
#define the model loss
def pearson_loss(x,y):
        mx = torch.mean(x, dim=1, keepdim=True)
        my = torch.mean(y, dim=1, keepdim=True)
        xm, ym = x - mx, y - my
    
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        loss = torch.sum(1-cos(xm,ym))
        return loss   

#define pearson loss with social regularization
def pearson_reg_loss(x,y, alpha):
        #load graph of cell lineages
        neighbors = np.load("../data/cell_graph.npy")
        neighbors = neighbors[~np.isnan(neighbors).any(axis=1), :]

        #compute loss
        mx = torch.mean(x, dim=1, keepdim=True)
        my = torch.mean(y, dim=1, keepdim=True)
        xm, ym = x - mx, y - my
    
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        loss = torch.sum(1-cos(xm,ym))
        print(loss)

        #scale centered prediction vector by standard deviation
        std = torch.std(x, dim=1).view(-1, 1).repeat(1, 81)
        xm = torch.div(xm, std)

        for edge in neighbors:
            neighbor1 = int(edge[0])
            neighbor2 = int(edge[1])
            diff = xm[:, neighbor1] - xm[:, neighbor2]
            loss += alpha * torch.sum(torch.mul(diff, diff))

        print(loss)
        return loss   
    

def train_model(train_loader, test_loader, model, device, criterion, optimizer, num_epochs, output_directory):
    total_step = len(train_loader)
    model.train()

    #open files to log error
    train_error = open(output_directory + "training_error.txt", "a")
    test_error = open(output_directory + "test_error.txt", "a")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss_valid = float('inf')
    best_epoch = 1

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (seqs, labels) in enumerate(train_loader):
            seqs = seqs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs, act, idx = model(seqs)
            loss = criterion(outputs, labels) # change input to 
            running_loss += loss.item()
        
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #if (i+1) % 100 == 0:
            #    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
            #           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        #save training loss to file
        epoch_loss = running_loss / len(train_loader.dataset)
        print("%s, %s" % (epoch, epoch_loss), file=train_error)

        #calculate test loss for epoch
        test_loss = 0.0
        with torch.no_grad():
            model.eval()
            for i, (seqs, labels) in enumerate(test_loader):
                x = seqs.to(device)
                y = labels.to(device)
                outputs, act, idx = model(x)
                loss = criterion(outputs, y)
                test_loss += loss.item() 

        test_loss = test_loss / len(test_loader.dataset)

        #save outputs for epoch
        print("%s, %s" % (epoch, test_loss), file=test_error)

        if test_loss < best_loss_valid:
            best_loss_valid = test_loss
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            print ('Saving the best model weights at Epoch [{}], Best Valid Loss: {:.4f}' 
                       .format(epoch+1, best_loss_valid))


    train_error.close()
    test_error.close()

    #model.load_state_dict(best_model_wts)
    return model, best_loss_valid
    

def test_model(test_loader, model, device):
    predictions = torch.zeros(0, 81)
    max_activations = torch.zeros(0, 300)
    act_index = torch.zeros(0, 300)

    with torch.no_grad():
        model.eval()
        for seqs, labels in test_loader:
            seqs = seqs.to(device)
            pred, act, idx = model(seqs)
            predictions = torch.cat((predictions, pred.type(torch.FloatTensor)), 0)
            max_activations = torch.cat((max_activations, act.type(torch.FloatTensor)), 0)
            act_index = torch.cat((act_index, idx.type(torch.FloatTensor)), 0)

    predictions = predictions.numpy()
    max_activations = max_activations.numpy()
    act_index = act_index.numpy()
    return predictions, max_activations, act_index



def get_motifs(data_loader, model, device):
    activations = torch.zeros(0, 300, 251)
    predictions = torch.zeros(0, 300, 81)
    with torch.no_grad():
        model.eval()
        for seqs, labels in data_loader:
            seqs = seqs.to(device)
            pred, act, idx = model(seqs)
            
            activations = torch.cat((activations, act.type(torch.FloatTensor)), 0)
            predictions = torch.cat((predictions, pred.type(torch.FloatTensor)), 0)
            
    predictions = predictions.numpy()
    activations = activations.numpy()
    return activations, predictions
