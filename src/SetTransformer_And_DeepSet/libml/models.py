from src.SetTransformer_And_DeepSet.libml.modules import *

class DeepSet(nn.Module):
    def __init__(self, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        
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
        
        self.enc = nn.Sequential(
                nn.Linear(200 * 4 * 4, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))

    def forward(self, x):
        
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        H = H.view(-1, 200 * 4 * 4) #when using the same Echo_data.py as our ABMIL Reg model, H is shape: [num_instance, feature_dim]
        
#         print('Inside DeepSet, before unsqueeze, H shape: {}'.format(H.shape))
        #self.enc expect input of shape: [batch_size, num_instance, feature_dim] 
        H = H.unsqueeze(0)
#         print('Inside DeepSet, after unsqueeze, H shape: {}'.format(H.shape))

        encoded_x = self.enc(H).mean(-2)
        decoded_x = self.dec(encoded_x).reshape(-1, self.num_outputs, self.dim_output)

        return decoded_x

class SetTransformer(nn.Module):
    def __init__(self, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        
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
        
        self.enc = nn.Sequential(
                ISAB(200 * 4 * 4, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, x):
        
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        H = H.view(-1, 200 * 4 * 4) #when using the same Echo_data.py as our ABMIL Reg model, H is shape: [num_instance, feature_dim]
        
#         print('Inside SetTransformer, before unsqueeze, H shape: {}'.format(H.shape))
        #self.enc expect input of shape: [batch_size, num_instance, feature_dim] 
        H = H.unsqueeze(0)
#         print('Inside SetTransformer, after unsqueeze, H shape: {}'.format(H.shape))

        encoded_x = self.enc(H)
        decoded_x = self.dec(encoded_x)
        
        return decoded_x
