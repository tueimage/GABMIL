import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
 

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
        

"""Combined local & global"""
    
class MLP(nn.Module):
    def __init__(self, num_features, expansion_factor):
        super().__init__()
        num_hidden = int(num_features * expansion_factor)
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.dropout(self.fc1(x)))
        return x

def window_partition(input, window_size=(7, 7)):
    """ Window partition function.
    Args:
        input (torch.Tensor): Input tensor of the shape [H, W, C].
        window_size (Tuple[int, int], optional): Window size to be applied. Default (7, 7)
    Returns:
        windows (torch.Tensor): Unfolded input tensor of the shape [windows, window_size[0], window_size[1], C].
    """
    # Get size of input
    H, W, C = input.shape    
    # Unfold input
    windows = input.view(H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    # Permute and reshape to [windows, window_size[0], window_size[1], channels]
    windows = windows.permute(0, 2, 1, 3, 4).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows

def window_reverse(windows, original_size, window_size= (7, 7)):
    """ Reverses the window partition.
    Args:
        windows (torch.Tensor): Window tensor of the shape [windows, window_size[0], window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)
    Returns:
        output (torch.Tensor): Folded output tensor of the shape [original_size[0], original_size[1], C].
    """
    # Get height and width
    H, W = original_size
    # Fold grid tensor
    output = windows.view(H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 2, 1, 3, 4).contiguous().view(H, W, -1)
    return output



def grid_partition(input, grid_size= (7, 7)):
    """ Grid partition function.

    Args:
        input (torch.Tensor): Input tensor of the shape [H, W, C].
        grid_size (Tuple[int, int], optional): Grid size to be applied. Default (7, 7)

    Returns:
        grid (torch.Tensor): Unfolded input tensor of the shape [grids, grid_size[0], grid_size[1], C].
    """
    # Get size of input
    H, W, C = input.shape
    # Unfold input
    grid = input.view(grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)
    # Permute and reshape [(H // grid_size[0]) * (W // grid_size[1]), grid_size[0], grid_size[1], C]
    grid = grid.permute(1, 3, 0, 2, 4).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return grid

def grid_reverse(grid, original_size, grid_size= (7, 7)):
    """ Reverses the grid partition.

    Args:
        Grid (torch.Tensor): Grid tensor of the shape [grids, grid_size[0], grid_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        grid_size (Tuple[int, int], optional): Grid size which have been applied. Default (7, 7)

    Returns:
        output (torch.Tensor): Folded output tensor of the shape [C, original_size[0], original_size[1]].
    """
    # Get height, width, and channels
    (H, W), C = original_size, grid.shape[-1]
    # Fold grid tensor
    output = grid.view(H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    output = output.permute(2, 0, 3, 1, 4).contiguous().view(H, W, C)
    return output



"""
args:
    use_local: whether to use local information
    use_grid: whether to use grid information
    use_block: whether to use block information
    n_classes: number of classes
    win_size_b: block window size
    win_size_g: grid window size
    magnification: magnification factor
"""
class GABMIL(nn.Module):
    def __init__(self, use_local= True, use_grid= False, use_block= False, n_classes=2, win_size_b= 1, win_size_g= 1, magnification= 10):
        super(GABMIL, self).__init__()
        self.use_local= use_local
        self.use_grid= use_grid
        self.use_block= use_block
        self.win_size_b= win_size_b
        self.win_size_g= win_size_g
        self.win_size = int(np.lcm(win_size_b, win_size_g))
        self.magnification= magnification
        size = [1024, 512, 256]
        
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = True, n_classes = 1)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.n_classes = n_classes
        
        if not self.use_local:
            if self.use_block:
                self.mlp1 = MLP(num_features= self.win_size_b*self.win_size_b, expansion_factor= 1)  
                self.window_partition = window_partition
                self.window_reverse = window_reverse
            if self.use_grid:
                self.mlp2 = MLP(num_features= self.win_size_g*self.win_size_g, expansion_factor= 1)  
                self.grid_partition = grid_partition
                self.grid_reverse = grid_reverse
    
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        if not self.use_local:
            if self.use_block:
                self.mlp1 = self.mlp1.to(device)
            if self.use_grid:    
                self.mlp2 = self.mlp2.to(device)
        
    def forward(self, h, coords= None, attention_only=False):
        device = h.device # Nxfeat_dim    
        _, C = h.shape

        "input data with preserved spatial information"
        scale = 40//self.magnification
        ref_patch_size = 256*scale
        min_x= coords[:, 0].min()//ref_patch_size 
        max_x= coords[:, 0].max()//ref_patch_size
        min_y= coords[:, 1].min()//ref_patch_size
        max_y= coords[:, 1].max()//ref_patch_size

        zz= torch.zeros((max_y-min_y+1, max_x-min_x+1,  C), device= device) # XxYxfeat_dim 
        coords[:, :] //= ref_patch_size
        cc_x= coords[:, 0]-min_x
        cc_y= coords[:, 1]-min_y
        zz[cc_y, cc_x]= h
    
        """Global"""
        X, Y, C = zz.shape
        _S_x = int(np.ceil(X/self.win_size))
        _S_y = int(np.ceil(Y/self.win_size)) 
        add_length_s_x = _S_x*self.win_size - X
        add_length_s_y = _S_y*self.win_size- Y
        zz_x = torch.zeros((add_length_s_x, Y, C), device= device)
        zz = torch.cat([zz, zz_x],dim = 0) # (X+x)xYxfeat_dim  
        zz_y = torch.zeros((X+add_length_s_x ,add_length_s_y, C), device= device)
        zz = torch.cat([zz, zz_y],dim = 1) # (X+x)x(Y+y)xfeat_dim  
                    
        H, W, C = zz.shape

        "Window"
        if self.use_block:
            windows= self.window_partition(zz, (self.win_size_b, self.win_size_b)) # [windows,  window_size[0], window_size[1], feat_dim]
            windows = windows.view(-1, self.win_size_b*self.win_size_b, C) # [windows, window_size[0]*window_size[1], feat_dim]
            windows = windows.permute(0, 2, 1) #  [windows, feat_dim, window_size[0]*window_size[1]]
            h_g_w = windows
            h_g_w = self.mlp1(h_g_w) # [windows, feat_dim, window_size[0]*window_size[1]]
            
            h_g_w = h_g_w.permute(0, 2, 1) # [windows, window_size[0]*window_size[1], feat_dim]
            h_g_w = h_g_w.view(-1, self.win_size_b, self.win_size_b, C) # [windows,  window_size[0], window_size[1], feat_dim]
            h_g_w = self.window_reverse(h_g_w, (H, W), (self.win_size_b, self.win_size_b)) # [original_size[0], original_size[1], feat_dim]
        else:
            h_g_w = zz
        
        "Grid"
        if self.use_grid:
            if self.use_block:
                h_g_w= h_g_w + zz
            grids= self.grid_partition(h_g_w, (self.win_size_g, self.win_size_g)) # [grids,  grid_size[0], grid_size[1], feat_dim]
            grids = grids.view(-1, self.win_size_g*self.win_size_g, C) # [grids, grid_size[0]*grid_size[1], feat_dim]
            grids = grids.permute(0, 2, 1) #  [grids, feat_dim, grid_size[0]*grid_size[1]]
            h_g_g = grids
            h_g_g = self.mlp2(h_g_g) # [grids, feat_dim, grid_size[0]*grid_size[1]]

            h_g_g = h_g_g.permute(0, 2, 1) # [grids, grid_size[0]*grid_size[1], feat_dim]
            h_g_g = h_g_g.view(-1, self.win_size_g, self.win_size_g, C) # [grids,  grid_size[0], grid_size[1], feat_dim]
            h_g_g = self.grid_reverse(h_g_g, (H, W), (self.win_size_g, self.win_size_g)) # [original_size[0], original_size[1], feat_dim]    
        else:
            h_g_g = h_g_w

        """Residual Connection"""
        if self.use_local:
            h = zz
        elif self.use_block:
            if self.use_grid:
                h= h_g_w + h_g_g   
            else:
                h= zz + h_g_g
        elif self.use_grid:
            h= zz + h_g_g

        "discard padding"
        h = h[cc_y, cc_x] 
        h= h.view(-1, C)
        
        """Local"""
        A, h = self.attention_net(h)  # NxK  
              
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        
        A = F.softmax(A, dim=1)  # softmax over N
        
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        
        return logits, Y_prob, Y_hat, A_raw