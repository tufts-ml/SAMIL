import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# class FCLayer(nn.Module):
#     def __init__(self, in_size, out_size=1):
#         super(FCLayer, self).__init__()
#         self.fc = nn.Sequential(nn.Linear(in_size, out_size))
#     def forward(self, feats):
#         x = self.fc(feats)
#         return feats, x

class IClassifier(nn.Module):
    def __init__(self,  num_classes=3):
        super(IClassifier, self).__init__()
        
        self.L = 500
        self.B = 250
        self.num_classes = num_classes
        
        self.feature_extractor_part1 = nn.Sequential(
#             nn.Conv2d(1, 20, kernel_size=5),
            nn.Conv2d(3, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            #hz added
            nn.Conv2d(50, 100, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(100, 200, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        
        self.feature_extractor_part2 = nn.Sequential(
#             nn.Linear(50 * 4 * 4, self.L),
            nn.Linear(200 * 4 * 4, self.L),
            nn.ReLU(),
        )
        
#         self.feature_extractor_part3 = nn.Sequential(
            
#             nn.Linear(self.L, self.B),
#             nn.ReLU(),
#             nn.Linear(self.B, self.L),
#             nn.ReLU(),
#         )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.L, self.num_classes),
#             nn.Sigmoid()
        )
        
#         self.feature_extractor = feature_extractor      
#         self.fc = nn.Linear(feature_size, output_class)
        
        
    def forward(self, x):
        
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        H = H.view(-1, 200 * 4 * 4) #when using the same Echo_data.py as our ABMIL Reg model, H is shape: [num_instance, feature_dim]
        
        H = self.feature_extractor_part2(H)  # NxL
        
        
        c = self.classifier(H)  # N x C (C is number of classes)
#         print('Inside IClassifier, c shape: {}'.format(c.shape))
        
        return H, c
#         device = x.device
#         feats = self.feature_extractor(x) # N x K
#         c = self.fc(feats.view(feats.shape[0], -1)) # N x C
#         return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module):
    def __init__(self, args, num_classes=3, dropout_v=0.0, nonlinear=True, passing_v=False): # K, L, N
        super(BClassifier, self).__init__()
        
        self.L = 500
        self.device = args.device
        
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(self.L, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(self.L, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(self.L, self.L),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(num_classes, num_classes, kernel_size=self.L)  
        
    def forward(self, feats, c): # N x K, N x C
#         device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
#         print('Inside BClassifier, m_indices shape: {}'.format(m_indices.shape))
        
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
#         print('Inside BClassifier, m_feat shape: {}'.format(m_feats.shape))
        
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
#         print('Inside BClassifier, q_max shape: {}'.format(q_max.shape))

        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
#         print('Inside BClassifier, A shape: {}'.format(A.shape))
        
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=self.device)), 0) # normalize attention scores, A in shape N x C, 
#         print('Inside BClassifier, A shape: {}'.format(A.shape))

        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
#         print('Inside BClassifier, B shape: {}'.format(B.shape))
  
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
#         print('Inside BClassifier, B shape: {}'.format(B.shape))

        C = self.fcc(B) # 1 x C x 1
#         print('Inside BClassifier, C shape: {}'.format(C.shape))

        C = C.view(1, -1)
#         print('Inside BClassifier, C shape: {}'.format(C.shape))

        return C, A, B 
    
class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        
    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        
        return classes, prediction_bag, A, B