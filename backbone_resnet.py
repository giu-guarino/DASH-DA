import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, resnet34


class FC_Classifier_NoLazy(torch.nn.Module):
    def __init__(self, input_dim, n_classes):
        super(FC_Classifier_NoLazy, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim,n_classes)
        )
    
    def forward(self, X):
        return self.block(X)


class DASH_DA(torch.nn.Module):
    def __init__(self, input_channel_source=4, input_channel_target=2, emb_dim = 256, num_classes=10):
        super(DASH_DA, self).__init__()

        source_model = resnet18(weights=None)
        source_model.conv1 = nn.Conv2d(input_channel_source, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.source = nn.Sequential(*list(source_model.children())[:-1])

        target_model = resnet18(weights=None)
        target_model.conv1 = nn.Conv2d(input_channel_target, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.target = nn.Sequential(*list(target_model.children())[:-1])

        self.domain_cl = FC_Classifier_NoLazy(emb_dim, 2)        
        self.task_cl = FC_Classifier_NoLazy(emb_dim, num_classes)        

    
    def forward_source(self, x, source):
        emb = None
        if source == 0:
            emb = self.source(x).squeeze()
        else:
            emb = self.target(x).squeeze()
        nfeat = emb.shape[1]
        emb_inv = emb[:,0:nfeat//2]
        emb_spec = emb[:,nfeat//2::]
        return emb_inv, emb_spec, self.domain_cl(emb_spec), self.task_cl(emb_inv)
        
    
    def forward(self, x):
        self.source.train()
        self.target.train()
        x_source, x_target = x
        emb_source_inv, emb_source_spec, dom_source_cl, task_source_cl = self.forward_source(x_source, 0)
        emb_target_inv, emb_target_spec, dom_target_cl, task_target_cl =  self.forward_source(x_target, 1)
        return emb_source_inv, emb_source_spec, dom_source_cl, task_source_cl, emb_target_inv, emb_target_spec, dom_target_cl, task_target_cl

    def forward_test_target(self, x):
        self.target.eval()
        emb_target_inv, emb_target_spec, dom_target_cl, task_target_cl =  self.forward_source(x, 1)
        return emb_target_inv, emb_target_spec, dom_target_cl, task_target_cl

    def forward_test_source(self, x):
        self.source.eval()
        emb_source_inv, emb_source_spec, dom_source_cl, task_source_cl = self.forward_source(x, 0)
        return emb_source_inv, emb_source_spec, dom_source_cl, task_source_cl

    def get_target_and_task_weights(self):
        # Return only the parameters, not the names
        target_params = list(self.target.parameters())
        task_cl_params = list(self.task_cl.parameters())
        return target_params, task_cl_params
