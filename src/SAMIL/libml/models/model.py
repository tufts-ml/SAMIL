import torch
import torch.nn as nn
import torch.nn.functional as F



class SAMIL(nn.Module):
    def __init__(self, num_classes=3):
        super(SAMIL, self).__init__()
        self.L = 500
        self.B = 250
        self.D = 128
        self.K = 1
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
        
        self.feature_extractor_part3 = nn.Sequential(
            
            nn.Linear(self.L, self.B),
            nn.ReLU(),
            nn.Linear(self.B, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

#         self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
#             nn.Linear(self.L*self.K, 1),
            nn.Linear(self.L*self.K, self.num_classes),
#             nn.Sigmoid()
        )

    def forward(self, x):
        
#         print('Inside forward: input x shape: {}'.format(x.shape))
        x = x.squeeze(0)
#         print('Inside forward: after squeeze x shape: {}'.format(x.shape))

        H = self.feature_extractor_part1(x)
#         print('Inside forward: after feature_extractor_part1 H shape: {}'.format(H.shape))
       
            
#         H = H.view(-1, 50 * 4 * 4)
        H = H.view(-1, 200 * 4 * 4)
#         print('Inside forward: after view H shape: {}'.format(H.shape))

        H = self.feature_extractor_part2(H)  # NxL
#         print('Inside forward: after feature_extractor_part2 H shape: {}'.format(H.shape))

        A_V = self.attention_V(H)  # NxK
#         print('Inside forward: A_V is {}, shape: {}'.format(A_V, A_V.shape))

        A_V = torch.transpose(A_V, 1, 0)  # KxN
#         print('Inside forward: A_V is {}, shape: {}'.format(A_V, A_V.shape))

        A_V = F.softmax(A_V, dim=1)  # softmax over N
#         print('Inside forward: A_V (View) is {}, shape: {}'.format(A_V, A_V.shape))
        
    
        H = self.feature_extractor_part3(H)
    
        A_U = self.attention_U(H)  # NxK
#         print('Inside forward: A_U is {}, shape: {}'.format(A_U, A_U.shape))

        A_U = torch.transpose(A_U, 1, 0)  # KxN
#         print('Inside forward: A_U is {}, shape: {}'.format(A_U, A_U.shape))

        A_U = F.softmax(A_U, dim=1)  # softmax over N
#         print('Inside forward: A_U (Diagnosis) is {}, shape: {}'.format(A_U, A_U.shape))
        
#         A = A_V * A_U
#         print('Inside forward: final A is {}, shape: {}'.format(A, A.shape))
        A = torch.exp(torch.log(A_V) + torch.log(A_U)) #numerically more stable?

        A = A/torch.sum(A)
#         A = F.softmax(A, dim=1)
#         print('Inside forward: final A is {}, shape: {}'.format(A, A.shape))
#         A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
#         print('Inside forward: A is {}, shape: {}'.format(A, A.shape))

#         A = torch.transpose(A, 1, 0)  # KxN
# #         print('Inside forward: A is {}, shape: {}'.format(A, A.shape))

#         A = F.softmax(A, dim=1)  # softmax over N
# #         print('Inside forward: A is {}, shape: {}'.format(A, A.shape))

        M = torch.mm(A, H)  # KxL #M can be regarded as final representation of this bag
#         print('Inside forward: M is {}, shape: {}'.format(M, M.shape))

        out = self.classifier(M)
        

        return out, A_V #only view regularize one branch of the attention weights

    