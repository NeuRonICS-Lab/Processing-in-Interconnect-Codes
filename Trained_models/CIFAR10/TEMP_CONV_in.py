import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_random_elements(tensor, drop_prob=0.3):
    """Randomly drops elements from a tensor with probability `drop_prob`."""
    mask = torch.rand(tensor.shape) > drop_prob  # Create a random mask
    return tensor[mask]

class K_layer_opt1s(torch.nn.Module):
    def __init__(self, input_dim, output_dim, k_in, k_act, diff):
        super().__init__()
        self.k_in = k_in
        self.k_act = k_act
        self.diff = diff
        torch.manual_seed(45)
        self.weight = torch.nn.Parameter(torch.empty(input_dim, output_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.weight, gain=1.0)
        self.weight.data = torch.clamp(self.weight.data, -3, 3)  # In-place clamping of weight values

    def spikeK(self, sorted_in: torch.Tensor):
      return (sorted_in.sum(dim=2)/(self.k_act)) #avg of min K values

    def spikeK1(self, sorted_in: torch.Tensor):
      thr,_ = torch.topk(sorted_in, self.k_act, dim=2, largest=False, sorted=False)
      sum_nonzero = thr.sum(dim=2)
      return (sum_nonzero/(self.k_act)) #avg of min K values

    def forward(self, input, inputn=None):
        input = torch.unsqueeze(input,axis=-1)
        inputn = F.relu((3-input))
        # print("input",input.size())
        Wp = F.relu(3+self.weight)
        Wn = F.relu(3-self.weight)
        s = inputn.size()[2]
        inn,innq = torch.topk(inputn,int(s*self.k_in),dim=2, largest=False, sorted=False)
        # inn,innq = torch.sort(inputn,dim=2,stable=True)
        # inn = inn[:,:, :self.k_in, :]
        # innq = innq[:,:, :self.k_in, :]

        innq = innq.squeeze(-1)
        zPlus = self.spikeK1(inn + Wn[innq,:])
        zMinus = self.spikeK1(inn + Wp[innq,:])

        return zPlus, zMinus
 
class temp_conv_k_opt1s(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k_in, k_act, kernel_size=3, dilation=1, padding=1, stride=1, diff=0):
        super(temp_conv_k_opt1s, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = (dilation, dilation)
        self.padding = padding
        self.stride = stride
        self.k_in = k_in
        self.k_act = k_act
        self.diff = diff
        self.cnn = K_layer_opt1s(self.in_channels * self.kernel_size * self.kernel_size, self.out_channels, self.k_in, self.k_act, diff=self.diff)
 
    def forward(self, inputs,inputn=None):
        inp_size = inputs.size()
        inp_unf_p = F.unfold(inputs, (self.kernel_size, self.kernel_size), stride=self.stride, padding=self.padding)
        patches = inp_unf_p.permute(0, 2, 1)  # Shape: [batch_size, num_patches, patch_size]
        if(inputn is not None):
          inp_unf_n = F.unfold(inputn, (self.kernel_size, self.kernel_size), stride=self.stride, padding=self.padding)
          patches_n = inp_unf_n.permute(0, 2, 1)  # Shape: [batch_size, num_patches, patch_size]
          output_p, output_n = self.cnn(patches,patches_n)
        else:
          output_p, output_n = self.cnn(patches)
        out_height = math.floor((inp_size[2] - self.kernel_size + 2 * self.padding) / self.stride) + 1 #compute new output height
        out_width = math.floor((inp_size[3] - self.kernel_size + 2 * self.padding) / self.stride) + 1 #compute new output width
        output_p = torch.reshape(output_p, (inp_size[0], out_height, out_width, self.out_channels))
        output_p = output_p.permute(0, 3, 1, 2)
        output_n = torch.reshape(output_n, (inp_size[0], out_height, out_width, self.out_channels))
        output_n = output_n.permute(0, 3, 1, 2)
        if(self.diff==0):    ##changes made here
          return output_n - output_p
        else:
          return output_p, output_n
         
        
class K_layer_opt1single(torch.nn.Module):
    def __init__(self, input_dim, output_dim, k_act, k_in=0, diff=0):
        super().__init__()
        self.k_in = k_in
        self.k_act = k_act
        self.diff = diff
        torch.manual_seed(45)
        self.weight = torch.nn.Parameter(torch.empty(input_dim, output_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.weight, gain=1.0)
        self.weight.data = torch.clamp(self.weight.data, -3, 3)  # In-place clamping of weight values

    def spikeK(self, sorted_in: torch.Tensor):
      return (sorted_in.sum(dim=2)/(self.k_act)) #avg of min K values
    
    def spikeK1(self, sorted_in: torch.Tensor):
      thr,_ = torch.topk(sorted_in, self.k_act, dim=2, largest=False, sorted=False)
      sum_nonzero = thr.sum(dim=2)
      return (sum_nonzero/(self.k_act)) #avg of min K values
    
    def forward(self, input, inputn=None):
        input = torch.unsqueeze(input,axis=-1)
        input = F.relu((3-input))
        Wp = F.relu(3+self.weight)
        Wn = F.relu(3-self.weight)
        zPlus = self.spikeK1(input + Wn)
        zMinus = self.spikeK1(input + Wp)
        return zPlus, zMinus
 
class temp_conv_k_opt1single(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k_act, k_in=0, kernel_size=3, dilation=1, padding=0, stride=1, diff=0):
        super(temp_conv_k_opt1single, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = (dilation, dilation)
        self.padding = padding
        self.stride = stride
        self.k_in = k_in
        self.k_act = k_act
        self.diff = diff
        self.cnn = K_layer_opt1single(self.in_channels * self.kernel_size * self.kernel_size, self.out_channels, self.k_act, self.k_in, diff=self.diff)
 
    def forward(self, inputs,inputn=None):
        inp_size = inputs.size()
        inp_unf_p = F.unfold(inputs, (self.kernel_size, self.kernel_size), stride=self.stride, padding=self.padding)
        patches = inp_unf_p.permute(0, 2, 1)  # Shape: [batch_size, num_patches, patch_size]
        if(inputn is not None):
          inp_unf_n = F.unfold(inputn, (self.kernel_size, self.kernel_size), stride=self.stride, padding=self.padding)
          patches_n = inp_unf_n.permute(0, 2, 1)  # Shape: [batch_size, num_patches, patch_size]
          output_p, output_n = self.cnn(patches,patches_n)
        else:
          output_p, output_n = self.cnn(patches)
        out_height = math.floor((inp_size[2] - self.kernel_size + 2 * self.padding) / self.stride) + 1 #compute new output height
        out_width = math.floor((inp_size[3] - self.kernel_size + 2 * self.padding) / self.stride) + 1 #compute new output width
        output_p = torch.reshape(output_p, (inp_size[0], out_height, out_width, self.out_channels))
        output_p = output_p.permute(0, 3, 1, 2)
        output_n = torch.reshape(output_n, (inp_size[0], out_height, out_width, self.out_channels))
        output_n = output_n.permute(0, 3, 1, 2)
        if(self.diff==0):    ##changes made here
          return output_n - output_p
        else:
          return output_p, output_n
        
#version of ConvK layer with reduced memory comsumption, larger latency
class MPLayer_K(torch.nn.Module):
  def __init__(self, weight, in_size,output_dim, gamma,diff=0):
      super().__init__()
      self.output_dim = output_dim
      self.gamma = gamma
      self.weight = weight
      self.diff = diff
      self.in_size = in_size

  def spikeK(self, sorted_in: torch.Tensor, gamma: float):
      if gamma == 0 or gamma == 1:
          out = torch.kthvalue(sorted_in, 1, dim=1).values
          return out
      thr,_ = torch.topk(sorted_in, gamma, dim=1, largest=False, sorted=False)
      sum_nonzero = thr.sum(dim=1)
      return (sum_nonzero/(gamma)) #avg of min K values

  def forward(self, input, inputn=None):
      input = torch.unsqueeze(input,axis=-1)
      if(self.diff==0):
        inputp = F.relu((3+input))
        inputn = F.relu((3-input))
      else:
        inputp = input
        inputn = torch.unsqueeze(inputn,axis=-1)

      Wp = F.relu(self.weight)
      Wn = F.relu(-self.weight)

      zpp,_ = torch.topk((inputp + Wp), self.gamma, dim=1, largest=False, sorted=False)
      znp,_ = torch.topk((inputn + Wn), self.gamma, dim=1, largest=False, sorted=False)
      zpn,_ = torch.topk((inputp + Wn), self.gamma, dim=1, largest=False, sorted=False)
      znn,_ = torch.topk((inputn + Wp), self.gamma, dim=1, largest=False, sorted=False)
      
      zP = torch.cat([zpp,znp], axis=1)
      zN = torch.cat([zpn,znn], axis=1)
       
      tzP = self.spikeK(zP, self.gamma)
      tzN = self.spikeK(zN, self.gamma)

      return tzP,tzN


class Temp_conv_K(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gamma, dilation=1, padding=1, stride=1,diff = 0):
        super(Temp_conv_K, self).__init__()

        self.kernel_size = kernel_size
        self.kernal_size_number = kernel_size * kernel_size
        self.out_channels = out_channels
        self.dilation = (dilation, dilation)
        self.padding = padding
        self.stride = stride
        self.in_channels = in_channels
        self.diff = diff
        torch.manual_seed(45)

        # Initialize the kernel
        self.kernel = torch.nn.Parameter(
            torch.empty(self.in_channels * self.kernel_size * self.kernel_size, self.out_channels),
            requires_grad=True
        )
        torch.nn.init.xavier_normal_(self.kernel, gain=1.0)
        self.kernel.data = torch.clamp(self.kernel.data, -3, 3)  # In-place clamping of kernel values
        self.gamma = gamma


    def forward(self, xp,xn=0):
        inp_size = xp.size() # Shape: [batch_size, in_channels, height, width]
        # Unfold the input tensor into patches
        if(self.diff==0):
          # with torch.no_grad():
              inp_unf = F.unfold(xp, (self.kernel_size, self.kernel_size), stride=self.stride, padding=self.padding) # Shape: [batch_size, in_channels, kernel_size^2]
              patchesp = inp_unf.permute(0, 2, 1)  # Shape: [batch_size, num_patches, patch_size]
              
        else:
          # with torch.no_grad():
              inp_unf = F.unfold(xp, (self.kernel_size, self.kernel_size), stride=self.stride, padding=self.padding) # Shape: [batch_size, in_channels, kernel_size^2]
              patchesp = inp_unf.permute(0, 2, 1)  # Shape: [batch_size, num_patches, patch_size]
              inn_unf = F.unfold(xn, (self.kernel_size, self.kernel_size), stride=self.stride, padding=self.padding) # Shape: [batch_size, in_channels, kernel_size^2]
              patchesn = inn_unf.permute(0, 2, 1)  # Shape: [batch_size, num_patches, patch_size]
              # del inp_unf, inn_unf  # Free up memory
   
        hidlayer = MPLayer_K(self.kernel,self.in_channels, self.out_channels, self.gamma,self.diff)# weight, in_size,output_dim, gamma,diff=0
        outputsp = []
        outputsn = []
        for i in range(patchesp.size(0)):
          if(self.diff==0):
            outputp, outputn = hidlayer(patchesp[i])
          else:
            outputp, outputn = hidlayer(patchesp[i],patchesn[i])
          outputsp.append(outputp)
          outputsn.append(outputn)

        # Combine and reshape outputs
        outputsp = torch.stack(outputsp)
        outputsn = torch.stack(outputsn)

        out_height = math.floor((inp_size[2] - self.kernel_size + 2 * self.padding) / self.stride) + 1 #compute new output height
        out_width = math.floor((inp_size[3] - self.kernel_size + 2 * self.padding) / self.stride) + 1 #compute new output width

        img_reshapedp = torch.reshape(outputsp, (inp_size[0], out_height, out_width, self.out_channels))
        img_reshapedp = img_reshapedp.permute(0, 3, 1, 2)  # Shape: [batch_size, out_channels, height, width]
        img_reshapedn = torch.reshape(outputsn, (inp_size[0], out_height, out_width, self.out_channels))
        img_reshapedn = img_reshapedn.permute(0, 3, 1, 2)  # Shape: [batch_size, out_channels, height, width]

        return img_reshapedp, img_reshapedn

      
      
class MPLayer_TEMP(torch.nn.Module):
  def __init__(self, weight, output_dim, gamma,diff=0):
      super().__init__()
      self.output_dim = output_dim
      self.gamma = gamma
      self.weight = weight
      self.diff = diff

  def spikeMP(self, sorted_in: torch.Tensor, gamma: float):
        device = sorted_in.device
        if(gamma==0):
           sorted_in, _= torch.kthvalue(sorted_in, 1, dim=1)
           del _
           return sorted_in

        sorted_in, _ = sorted_in.sort(dim=1) #sort input (n elements)
        diff = (sorted_in[:, 1:, :] - sorted_in[:, :-1, :]) #find cumulative diff between consequtive elements (n-1) elements
        diff.mul_(torch.arange(sorted_in.shape[1]-1,device=device).unsqueeze(0).unsqueeze(2) + 1) #multiple the diff by slopes of 1,2,3
        diff = torch.cumsum(diff, dim=1) #find cumulative sum
        diff.lt_(gamma) #find which sum is less than gamma
        diff = torch.cat((torch.ones((sorted_in.shape[0], 1, sorted_in.shape[2]),device=device), diff), dim=1)#add 1 to the left to make n elements from n-1
        diff.mul_(sorted_in)#find the relevant inputs that contribute
        sum = torch.sum(diff, dim=1) #sum the value
        sum.add_(gamma) #add gamma
        nonzero = torch.count_nonzero(diff, dim=1) #find the no of non zero elements
        sum.div_(torch.clamp(nonzero, min=1.0))  #find avg
        del diff, sorted_in, _
        torch.cuda.empty_cache()
        return sum

  def forward(self, inputp, inputn=None):
      inputp = torch.unsqueeze(inputp,axis=-1)
      if(self.diff==0):
        plusIn = F.relu((3+inputp))
        minusIn = F.relu((3-inputp))
      else:
        minusIn = torch.unsqueeze(inputn,axis=-1)
        plusIn = inputp

      plusW = F.relu(self.weight)
      minusW = F.relu(-self.weight)

      zPlus = torch.cat([(plusIn+plusW),(minusIn+minusW)],axis=1)
      zMinus = torch.cat([(plusIn+minusW),(minusIn+plusW)],axis=1)

      zPlus = self.spikeMP(zPlus, self.gamma)
      zMinus = self.spikeMP(zMinus, self.gamma)
      torch.cuda.empty_cache()
      return zPlus,zMinus


class Temp_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gamma, dilation=1, padding=0, stride=1,diff = 0):
        super(Temp_conv, self).__init__()
        self.kernel_size = kernel_size
        self.kernal_size_number = kernel_size * kernel_size
        self.out_channels = out_channels
        self.dilation = (dilation, dilation)
        self.padding = padding
        self.stride = stride
        self.in_channels = in_channels
        self.diff = diff
        torch.manual_seed(45)
        
        # Initialize the kernel
        self.kernel = torch.nn.Parameter(
            torch.empty(self.in_channels * self.kernel_size * self.kernel_size, self.out_channels),
            requires_grad=True
        )
        torch.nn.init.xavier_normal_(self.kernel, gain=1.0)
        self.kernel.data = torch.clamp(self.kernel.data, -3, 3)  # In-place clamping of kernel values
        self.gamma = gamma

    def forward(self, xp,xn=0):
        inp_size = xp.size() # Shape: [batch_size, in_channels, height, width]
        # Unfold the input tensor into patches
        if(self.diff==0):
          # with torch.no_grad():
              inp_unf = F.unfold(xp, (self.kernel_size, self.kernel_size), stride=self.stride, padding=self.padding) # Shape: [batch_size, in_channels, kernel_size^2]
              patchesp = inp_unf.permute(0, 2, 1)  # Shape: [batch_size, num_patches, patch_size]    
        else:
          # with torch.no_grad():
              inp_unf = F.unfold(xp, (self.kernel_size, self.kernel_size), stride=self.stride, padding=self.padding) # Shape: [batch_size, in_channels, kernel_size^2]
              patchesp = inp_unf.permute(0, 2, 1)  # Shape: [batch_size, num_patches, patch_size]
              inn_unf = F.unfold(xn, (self.kernel_size, self.kernel_size), stride=self.stride, padding=self.padding) # Shape: [batch_size, in_channels, kernel_size^2]
              patchesn = inn_unf.permute(0, 2, 1)  # Shape: [batch_size, num_patches, patch_size]
              # del inp_unf, inn_unf  # Free up memory
   
        hidlayer = MPLayer_TEMP(self.kernel, self.out_channels, self.gamma,self.diff)#weight, output_dim, gamma,diff=0
        outputsp = []
        outputsn = []
        for i in range(patchesp.size(0)):
          if(self.diff==0):
            outputp, outputn = hidlayer(patchesp[i])
          else:
            outputp, outputn = hidlayer(patchesp[i],patchesn[i])
          outputsp.append(outputp)
          outputsn.append(outputn)

        # Combine and reshape outputs
        outputsp = torch.stack(outputsp)
        outputsn = torch.stack(outputsn)

        out_height = math.floor((inp_size[2] - self.kernel_size + 2 * self.padding) / self.stride) + 1 #compute new output height
        out_width = math.floor((inp_size[3] - self.kernel_size + 2 * self.padding) / self.stride) + 1 #compute new output width

        img_reshapedp = torch.reshape(outputsp, (inp_size[0], out_height, out_width, self.out_channels))
        img_reshapedp = img_reshapedp.permute(0, 3, 1, 2)  # Shape: [batch_size, out_channels, height, width]
        img_reshapedn = torch.reshape(outputsn, (inp_size[0], out_height, out_width, self.out_channels))
        img_reshapedn = img_reshapedn.permute(0, 3, 1, 2)  # Shape: [batch_size, out_channels, height, width]

        return img_reshapedp, img_reshapedn

class K_layer_opt1(nn.Module):
    def __init__(self, input_dim, output_dim, gamma, diff):
        super().__init__()
        self.gamma = gamma
        self.diff = diff
        torch.manual_seed(45)
        self.weight = torch.nn.Parameter(torch.empty(input_dim, output_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.weight, gain=1.0)
        self.weight.data = torch.clamp(self.weight.data, -3, 3)  # In-place clamping of weight values    
    
    def spikeK(self, sorted_in: torch.Tensor, gamma: float):
      if gamma == 0 or gamma == 1:
          out = torch.kthvalue(sorted_in, 1, dim=2).values
          return out
      thr,_ = torch.topk(sorted_in, gamma, dim=2, largest=False, sorted=False)
      sum_nonzero = thr.sum(dim=2)
      return (sum_nonzero/(gamma)) #avg of min K values

    def forward(self, input, inputn=None):
        input = torch.unsqueeze(input,axis=-1)
        if inputn is not None:
            inputp = input
            inputn = torch.unsqueeze(inputn, axis=-1)
        else:
            inputp = F.relu((3 + input)) #remove F.relu
            inputn = F.relu((3 - input))

        Wp = F.relu(3+self.weight)
        Wn = F.relu(3-self.weight)
                  
        if(self.gamma > 0):
          zpp,_ = torch.topk((inputp + Wp), self.gamma, dim=2, largest=False, sorted=False)
          znp,_ = torch.topk((inputn + Wn), self.gamma, dim=2, largest=False, sorted=False)
          zpn,_ = torch.topk((inputp + Wn), self.gamma, dim=2, largest=False, sorted=False)
          znn,_ = torch.topk((inputn + Wp), self.gamma, dim=2, largest=False, sorted=False)
          del _
          torch.cuda.empty_cache()
          zP = torch.cat([zpp,znp], axis=2)
          zN = torch.cat([zpn,znn], axis=2)        
        else:
          zP = torch.cat([(inputp + Wp),(inputn + Wn)], axis=2)
          zN = torch.cat([(inputn + Wp),(inputp + Wn)], axis=2)       
        tzP = self.spikeK(zP, self.gamma)
        tzN = self.spikeK(zN, self.gamma)        
        return tzP, tzN
 
 
class temp_conv_k_opt1(nn.Module):
    def __init__(self, in_channels, out_channels, gamma,  kernel_size=3, dilation=1, padding=1, stride=1, diff=0):
        super(temp_conv_k_opt1, self).__init__() 
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = (dilation, dilation)
        self.padding = padding
        self.stride = stride
        self.gamma = gamma
        self.diff = diff
        self.cnn = K_layer_opt(self.in_channels * self.kernel_size * self.kernel_size, self.out_channels, self.gamma, diff=self.diff)        
    
    def forward(self, inputs,inputn=None): 
        inp_size = inputs.size()     
        inp_unf_p = F.unfold(inputs, (self.kernel_size, self.kernel_size), stride=self.stride, padding=self.padding)
        patches = inp_unf_p.permute(0, 2, 1)  # Shape: [batch_size, num_patches, patch_size]
        if(inputn is not None):
          inp_unf_n = F.unfold(inputn, (self.kernel_size, self.kernel_size), stride=self.stride, padding=self.padding)
          patches_n = inp_unf_n.permute(0, 2, 1)  # Shape: [batch_size, num_patches, patch_size]
          output_p, output_n = self.cnn(patches,patches_n)
        else:
          output_p, output_n = self.cnn(patches)
        out_height = math.floor((inp_size[2] - self.kernel_size + 2 * self.padding) / self.stride) + 1 #compute new output height
        out_width = math.floor((inp_size[3] - self.kernel_size + 2 * self.padding) / self.stride) + 1 #compute new output width
        output_p = torch.reshape(output_p, (inp_size[0], out_height, out_width, self.out_channels))
        output_p = output_p.permute(0, 3, 1, 2)
        output_n = torch.reshape(output_n, (inp_size[0], out_height, out_width, self.out_channels))
        output_n = output_n.permute(0, 3, 1, 2)
        if(self.diff==0):    ##changes made here
          return output_p - output_n
        else:
          return output_p, output_n 
    
class K_layer_opt(nn.Module):
    def __init__(self, input_dim, output_dim, gamma, diff,sparse,drop_prob=0.3):
        super().__init__()
        self.gamma = gamma
        self.diff = diff
        self.sparse = sparse
        self.drop_prob = drop_prob
        # torch.manual_seed(45)
        self.weight = torch.nn.Parameter(torch.empty(input_dim, output_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.weight, gain=1.0)
        self.weight.data = torch.clamp(self.weight.data, -3, 3)  # In-place clamping of weight values    
    
    def spikeK(self, sorted_in: torch.Tensor, gamma: float):
      if gamma == 0 or gamma == 1:
          out = torch.kthvalue(sorted_in, 1, dim=2).values
          return out
      thr,_ = torch.topk(sorted_in, gamma, dim=2, largest=False, sorted=False)
      sum_nonzero = thr.sum(dim=2)
      return (sum_nonzero/(gamma)) #avg of min K values

    def forward(self, input, inputn=None):
        input = torch.unsqueeze(input,axis=-1)
        if inputn is not None:
            inputp = input
            inputn = torch.unsqueeze(inputn, axis=-1)
        else:
            inputp = F.relu((3 + input)) #remove F.relu
            inputn = F.relu((3 - input))

        Wp = F.relu(3+self.weight)
        Wn = F.relu(3-self.weight)
        
        if(self.sparse):  
          a = inputp.size()[2]
          rand_idx = torch.randperm(a)
          drop_count = int(torch.round(torch.tensor(self.drop_prob * a)).item())
          rand_idx = rand_idx[:drop_count]
          inputp = inputp[:,:, rand_idx, :]
          inputn = inputn[:,:, rand_idx, :]
          Wp = (Wp[rand_idx,:])
          Wn = (Wn[rand_idx,:])
                  
        if(self.gamma > 0):
          zpp,_ = torch.topk((inputp + Wp), self.gamma, dim=2, largest=False, sorted=False)
          znp,_ = torch.topk((inputn + Wn), self.gamma, dim=2, largest=False, sorted=False)
          zpn,_ = torch.topk((inputp + Wn), self.gamma, dim=2, largest=False, sorted=False)
          znn,_ = torch.topk((inputn + Wp), self.gamma, dim=2, largest=False, sorted=False)
          del _
          zP = torch.cat([zpp,znp], axis=2)
          zN = torch.cat([zpn,znn], axis=2)        
        else:
          zP = torch.cat([(inputp + Wp),(inputn + Wn)], axis=2)
          zN = torch.cat([(inputn + Wp),(inputp + Wn)], axis=2)       
        tzP = self.spikeK(zP, self.gamma)
        tzN = self.spikeK(zN, self.gamma)        
        return tzP, tzN
 
 
class temp_conv_k_opt(nn.Module):
    def __init__(self, in_channels, out_channels, gamma,  kernel_size=3, dilation=1, padding=1, stride=1, diff=0,sparse = False,drop_prob=0.3):
        super(temp_conv_k_opt, self).__init__() 
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = (dilation, dilation)
        self.padding = padding
        self.stride = stride
        self.gamma = gamma
        self.diff = diff
        self.cnn = K_layer_opt(self.in_channels * self.kernel_size * self.kernel_size, self.out_channels, self.gamma, diff=self.diff,sparse=sparse,drop_prob=drop_prob)        
    
    def forward(self, inputs,inputn=None): 
        inp_size = inputs.size()     
        inp_unf_p = F.unfold(inputs, (self.kernel_size, self.kernel_size), stride=self.stride, padding=self.padding)
        patches = inp_unf_p.permute(0, 2, 1)  # Shape: [batch_size, num_patches, patch_size]
        if(inputn is not None):
          inp_unf_n = F.unfold(inputn, (self.kernel_size, self.kernel_size), stride=self.stride, padding=self.padding)
          patches_n = inp_unf_n.permute(0, 2, 1)  # Shape: [batch_size, num_patches, patch_size]
          output_p, output_n = self.cnn(patches,patches_n)
        else:
          output_p, output_n = self.cnn(patches)
        out_height = math.floor((inp_size[2] - self.kernel_size + 2 * self.padding) / self.stride) + 1 #compute new output height
        out_width = math.floor((inp_size[3] - self.kernel_size + 2 * self.padding) / self.stride) + 1 #compute new output width
        output_p = torch.reshape(output_p, (inp_size[0], out_height, out_width, self.out_channels))
        output_p = output_p.permute(0, 3, 1, 2)
        output_n = torch.reshape(output_n, (inp_size[0], out_height, out_width, self.out_channels))
        output_n = output_n.permute(0, 3, 1, 2)
        if(self.diff==0):    ##changes made here
          return output_p - output_n
        else:
          return output_p, output_n 
    

