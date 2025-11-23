import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler

#readme
#Define MPLayer_in_opt - TEMP input Layer 
#Define MPLayer_in_K - K input Layer 
#Define MPLayer_in_Ks - K sparse input Layer 
#Define MPLayer_in_org - unoptimized TEMP Layer 

class MPLayer_in_Ks(torch.nn.Module):
  def __init__(self,inp_node,out_node,k_in,k_act, diff=0,sparse=0,drop_prob=0.5):
    super().__init__()
    self.inp_node = inp_node
    self.out_node = out_node
    self.k_in = k_in
    self.k_act = k_act
    self.diff = diff
    torch.manual_seed(43)
    self.sparse = sparse
    self.drop_prob = drop_prob
    self.weight = torch.nn.Parameter(torch.empty(inp_node, out_node), requires_grad=True)
    torch.nn.init.xavier_normal_(self.weight, gain=1.0)
    torch.clamp(self.weight,-3,3)                                               # will it be clamped after each backward pass?
 
  def spikeK(self, sorted_in: torch.Tensor):
      return (sorted_in.sum(dim=1)/(self.k_act)) #avg of min K values

  def spikeK1(self, sorted_in: torch.Tensor):
      thr,_ = torch.topk(sorted_in, self.k_act, dim=1, largest=False, sorted=False)
      sum_nonzero = thr.sum(dim=1)
      return (sum_nonzero/(self.k_act)) #avg of min K values
        
  def forward(self, inputp):
      # inputp = inputp.permute(0,2,1)
      inputp = ((3-inputp))
      s = inputp.size()[1]

      inputp,innq = torch.topk(inputp, int(s*self.k_in), dim=1, largest=False, sorted=False)   
      # inputp,innq = torch.sort(inputp,stable=True,dim=1)
      inputp = torch.unsqueeze(inputp,axis=-1)
      # inputp = inputp[:, :self.k_in,:]
      # innq = innq[:, :self.k_in]
      innq = innq.squeeze(-1)
      zPlus = self.spikeK1(inputp +  F.relu(3-self.weight)[innq,:])
      zMinus = self.spikeK1(inputp + F.relu(3+self.weight) [innq,:])

      # torch.cuda.empty_cache()
      if(self.diff == 0):
        return (zMinus - zPlus)  ## previous TEMP based codes will not be compatible because of this change
      else:
        return zPlus,zMinus

class MPLayer_in_Ksingle(torch.nn.Module):
  def __init__(self,inp_node,out_node,k_act,k_in=0, diff=0,sparse=0,drop_prob=0.5):
    super().__init__()
    self.inp_node = inp_node
    self.out_node = out_node
    self.k_in = k_in
    self.k_act = k_act
    self.diff = diff
    torch.manual_seed(43)
    self.sparse = sparse
    self.drop_prob = drop_prob
    self.weight = torch.nn.Parameter(torch.empty(inp_node, out_node), requires_grad=True)
    torch.nn.init.xavier_normal_(self.weight, gain=1.0)
    torch.clamp(self.weight,-3,3)                                               # will it be clamped after each backward pass?
 
  def spikeK(self, sorted_in: torch.Tensor):
      return (sorted_in.sum(dim=1)/(self.k_act)) #avg of min K values

  def spikeK1(self, sorted_in: torch.Tensor):
      thr,_ = torch.topk(sorted_in, self.k_act, dim=1, largest=False, sorted=False)
      sum_nonzero = thr.sum(dim=1)
      return (sum_nonzero/(self.k_act)) #avg of min K values
        
  def forward(self, inputp):
      # inputp = inputp.permute(0,2,1)
      inputp = ((3-inputp))
      inputp = torch.unsqueeze(inputp,axis=-1)
      zPlus = self.spikeK1(inputp +  F.relu(3-self.weight)) #was initially interchanged
      zMinus = self.spikeK1(inputp + F.relu(3+self.weight))
      # torch.cuda.empty_cache()
      if(self.diff == 0):
        return (zMinus - zPlus)  ## previous TEMP based codes will not be compatible because of this change
      else:
        return zPlus,zMinus
            
class MPLayer_in_opt(torch.nn.Module):
  def __init__(self,inp_node,out_node,gamma,diff=0,sparse=0,drop_prob=0.5):
    super().__init__()
    self.inp_node = inp_node
    self.out_node = out_node
    self.gamma = gamma
    self.sparse = sparse
    self.drop_prob = drop_prob
    self.diff = diff # differential inputs are given or not
    torch.manual_seed(43)
    self.weight = torch.nn.Parameter(torch.empty(inp_node, out_node), requires_grad=True)
    torch.nn.init.xavier_normal_(self.weight, gain=1.0)
    torch.clamp(self.weight,-3,3)

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
      self.weight.type_as(inputp)
      if(inputn==None):
          # newInputs = inputp.repeat(1, 1, filters)
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
      if(self.diff == 0):
        return zPlus - zMinus  ## previous TEMP based codes will not be compatible because of this change
      else:
        return zPlus,zMinus
      
      
class MPLayer_in_K(torch.nn.Module):
  def __init__(self,inp_node,out_node,gamma,diff=0,sparse=0,drop_prob=0.5):
    super().__init__()
    self.inp_node = inp_node
    self.out_node = out_node
    self.gamma = gamma
    self.diff = diff # differential inputs are given or not
    # torch.manual_seed(43)
    self.sparse = sparse
    self.drop_prob = drop_prob
    self.weight = torch.nn.Parameter(torch.empty(inp_node, out_node), requires_grad=True)
    torch.nn.init.xavier_normal_(self.weight, gain=1.0)
    torch.clamp(self.weight,-3,3)

  def spikeK(self, sorted_in: torch.Tensor, gamma: float):
      if gamma == 0 or gamma == 1:
          out = torch.kthvalue(sorted_in, 1, dim=1).values
          return out
      thr,_ = torch.topk(sorted_in, gamma, dim=1, largest=False, sorted=False)
      sum_nonzero = thr.sum(dim=1)
      return (sum_nonzero/(gamma)) #avg of min K values

  def forward(self, inputp, inputn=None):
      inputp = torch.unsqueeze(inputp,axis=-1)
      self.weight.type_as(inputp)
      if(inputn==None):
        plusIn = ((3+inputp))
        minusIn = ((3-inputp))
      else:
        minusIn = torch.unsqueeze(inputn,axis=-1)
        plusIn = inputp
      
      plusW = F.relu(3+self.weight)
      minusW = F.relu(3-self.weight)
       
      if(self.sparse):
        rand_idx = torch.randperm(self.inp_node)
        l = torch.round(torch.tensor(self.drop_prob*self.inp_node)).item()
        rand_idx = rand_idx[:int(l)]
        plusIn = plusIn[:, rand_idx, :]
        minusIn = minusIn[:, rand_idx, :]
        plusW = (plusW[rand_idx,:])
        minusW = (minusW[rand_idx,:])
        
      zPlus = torch.cat([(plusIn+plusW),(minusIn+minusW)],axis=1)
      zMinus = torch.cat([(plusIn+minusW),(minusIn+plusW)],axis=1)
      
      zPlus = self.spikeK(zPlus, self.gamma)
      zMinus = self.spikeK(zMinus, self.gamma)
      torch.cuda.empty_cache()
      if(self.diff == 0):
        return zPlus - zMinus  ## previous TEMP based codes will not be compatible because of this change
      else:
        return zPlus,zMinus


class MPLayer_in_K1(torch.nn.Module):
  def __init__(self,inp_node,out_node,gamma,diff=0,sparse=0,drop_prob=0.5):
    super().__init__()
    self.inp_node = inp_node
    self.out_node = out_node
    self.gamma = gamma
    self.diff = diff # differential inputs are given or not
    torch.manual_seed(43)
    self.sparse = sparse
    self.drop_prob = drop_prob
    self.weight = torch.nn.Parameter(torch.empty(inp_node, out_node), requires_grad=True)
    torch.nn.init.xavier_normal_(self.weight, gain=1.0)
    torch.clamp(self.weight,-3,3)

  def spikeK(self, sorted_in: torch.Tensor, gamma: float):
      if gamma == 0 or gamma == 1:
          out = torch.kthvalue(sorted_in, 1, dim=1).values
          return out

      # thr = torch.kthvalue(sorted_in, gamma, dim=1).values #find the kth min value
      # mask = sorted_in < thr.unsqueeze(1) #find inputs lesser than the kth min value
      # sum_nonzero = (sorted_in * mask).sum(dim=1) #sum the min k inputs
      # del  mask
      thr,_ = torch.topk(sorted_in, gamma, dim=1, largest=False, sorted=False)
      sum_nonzero = thr.sum(dim=1)
      return (sum_nonzero/(gamma)) #avg of min K values

  def forward(self, inputp, inputn=None):
      inputp = torch.unsqueeze(inputp,axis=-1)
      self.weight.type_as(inputp)
      if(inputn==None):
        plusIn =  F.relu((3+inputp))
        minusIn = F.relu((3-inputp))
      else:
        minusIn = torch.unsqueeze(inputn,axis=-1)
        plusIn = inputp
        
      plusW = F.relu(3+self.weight)
      minusW = F.relu(3-self.weight)
        
      zPlus = torch.cat([(plusIn+plusW),(minusIn+minusW)],axis=1)
      zMinus = torch.cat([(plusIn+minusW),(minusIn+plusW)],axis=1)

      zPlus = self.spikeK(zPlus, self.gamma)
      zMinus = self.spikeK(zMinus, self.gamma)
      torch.cuda.empty_cache()
      if(self.diff == 0):
        return zPlus - zMinus  ## previous TEMP based codes will not be compatible because of this change
      else:
        return zPlus,zMinus

class MPLayer_in_alpha(torch.nn.Module):
  def __init__(self,inp_node,out_node,gamma,diff=0):
    super().__init__()
    self.inp_node = inp_node
    self.out_node = out_node
    self.gamma = gamma
    self.diff = diff # differential inputs are given or not
    torch.manual_seed(43)
    self.weight = torch.nn.Parameter(torch.empty(inp_node, out_node), requires_grad=True)
    torch.nn.init.xavier_normal_(self.weight, gain=1.0)
    torch.clamp(self.weight,-3,3)

  def spikeK(self, sorted_in: torch.Tensor, gamma: float):
      # if gamma == 0:
      #     out = torch.kthvalue(sorted_in, 1, dim=1).values
      #     return out

      thr = torch.kthvalue(sorted_in, 1, dim=1).values #find the kth min value
      thr = torch.mul(thr,(1+gamma))
      mask = sorted_in <= thr.unsqueeze(1) #find inputs lesser than the kth min value
      sum_nonzero = (sorted_in * mask).sum(dim=1) #sum the min k inputs
      count_nonzero = mask.sum(dim=1).float()
      return (sum_nonzero/(count_nonzero)),count_nonzero #avg of min K values

  def forward(self, inputp, inputn=None,beta=0):
      inputp = torch.unsqueeze(inputp,axis=-1)
      self.weight.type_as(inputp)
      if(inputn==None):
        plusIn = F.relu((3+inputp))
        minusIn = F.relu((3-inputp))
      else:
        minusIn = torch.unsqueeze(inputn,axis=-1)
        plusIn = inputp
        
      plusW = F.relu(3+self.weight)
      minusW = F.relu(3-self.weight)
     
      zPlus = torch.cat([(plusIn+plusW),(minusIn+minusW)],axis=1)
      zMinus = torch.cat([(plusIn+minusW),(minusIn+plusW)],axis=1)

      zPlus,cc = self.spikeK(zPlus, self.gamma+beta)
      zMinus,cc = self.spikeK(zMinus, self.gamma+beta)
      torch.cuda.empty_cache()
      if(self.diff == 0):
        return zPlus - zMinus,cc  ## previous TEMP based codes will not be compatible because of this change
      else:
        return zPlus,zMinus,cc



  



      

            
  
      
# class MPLayer_in_K(torch.nn.Module):
#   def __init__(self,inp_node,out_node,gamma,diff=0):
#     super().__init__()
#     self.inp_node = inp_node
#     self.out_node = out_node
#     self.gamma = gamma
#     self.diff = diff # differential inputs are given or not
#     torch.manual_seed(44)
#     self.weight = torch.nn.Parameter(torch.empty(inp_node, out_node), requires_grad=True)
#     torch.nn.init.xavier_normal_(self.weight, gain=1.0)
#     torch.clamp(self.weight,-3,3)

#   # @torch.jit.script
#   def spikeK(self, sorted_in: torch.Tensor, gamma: float):
#       if gamma == 0 or gamma == 1:
#           out = torch.kthvalue(sorted_in, 1, dim=1).values
#           return out

#       thr = torch.kthvalue(sorted_in, gamma, dim=1).values #find the kth min value
#       mask = sorted_in < thr.unsqueeze(1) #find inputs lesser than the kth min value
#       sum_nonzero = (sorted_in * mask).sum(dim=1) #sum the min k inputs
#       # count_nonzero = mask.sum(dim=1).float()
#       # avg_nonzero = sum_nonzero / torch.clamp(count_nonzero, min=1.0)
#       del  mask
#       return (sum_nonzero/(gamma)) #avg of min k-1 values

#   def forward(self, input, inputn=None):
#       input = torch.unsqueeze(input,axis=-1)
#       self.weight.type_as(input)
#       filters = self.weight.shape[1]
#       if(self.diff==0):
#         newInputs = input.repeat(1, 1, filters)
#         inputp = F.relu((3+newInputs))
#         inputn = F.relu((3-newInputs))
#       else:
#         inputn = torch.unsqueeze(inputn,axis=-1)
 
#       plusW = F.relu(+self.weight)
#       minusW = F.relu(-self.weight)

#       zpp,_ = torch.topk((inputp + plusW), self.gamma, dim=1, largest=False, sorted=False)
#       znp,_ = torch.topk((inputn + minusW), self.gamma, dim=1, largest=False, sorted=False)
#       zpn,_ = torch.topk((inputp + minusW), self.gamma, dim=1, largest=False, sorted=False)
#       znn,_ = torch.topk((inputn + plusW), self.gamma, dim=1, largest=False, sorted=False)
      
#       zPlus = torch.cat([zpp,znp], axis=2)
#       zMinus = torch.cat([zpn,znn], axis=2)
   
#       zPlus = self.spikeK(zPlus, self.gamma)
#       zMinus = self.spikeK(zMinus, self.gamma)
#       torch.cuda.empty_cache()
#       return zPlus,zMinus
    
#Define TEMP input Layer without diff relu single outputs
class MPLayer_in_org(torch.nn.Module):
  def __init__(self,inp_node,out_node,gamma):
    super().__init__()
    self.inp_node = inp_node
    self.out_node = out_node
    self.gamma = gamma
    torch.manual_seed(42)
    self.weight = torch.nn.Parameter(torch.empty(inp_node, out_node), requires_grad=True)
    torch.nn.init.xavier_normal_(self.weight, gain=1.0)
    torch.clamp(self.weight,-3,3)

  @torch.jit.script
  def spikeMP(device: torch.device, inMat: torch.Tensor, gamma: float):
        inMat,ind = torch.sort(inMat,dim=1)
        if(gamma==0):
           out,v = torch.min(inMat, dim=1)
           return out
        batch_num = inMat.shape[0]
        out_size =  inMat.shape[2]
        cs_t = torch.cumsum(inMat,dim=1)
        cs_t.add_(gamma)
        d = torch.ones_like(inMat)
        d.cumsum_(dim=1)
        cs_t.div_(d)
        
        row_to_add = 999*torch.ones([batch_num,1,out_size],dtype=torch.float32).to(device)
        arr_1 = torch.cat((row_to_add, cs_t), dim=1).to(device)
        row_to_add.zero_()
        arr_2 = torch.cat((inMat, row_to_add), dim=1).to(device)
        out = torch.where(arr_1 > arr_2, 999 * torch.ones_like(arr_1), arr_1).to(device)
        out,ind = torch.min(out, dim=1)
        out = torch.where(out == 999, arr_1[:,-1], out).to(device)
        return out

  def forward(self, inputs, device):
    inputs = torch.unsqueeze(inputs,axis=-1)
    filters = self.weight.shape[1]
    self.weight.type_as(inputs)
    for i in range(filters):
          if (i==0):
             newInputs = torch.cat([inputs],dim=-1)
          else:
             newInputs = torch.cat([newInputs,inputs],dim=-1)
            
    #during training
    plusIn = ((1+newInputs)+1)
    minusIn = ((1-newInputs)+1)    
    plusW = F.relu(self.weight)
    minusW = F.relu(-self.weight)
    plusXplusW = (plusIn+plusW)
    minusXminusW = (minusIn+minusW)
    plusXminusW = (plusIn+minusW)
    minusXplusW = (minusIn+plusW)
    zPlus = torch.cat([plusXplusW,minusXminusW],axis=1)
    zMinus = torch.cat([plusXminusW,minusXplusW],axis=1)

    zPlus = self.spikeMP(device,zPlus, self.gamma)
    zMinus = self.spikeMP(device, zMinus,self.gamma)

    return zPlus,zMinus


 
