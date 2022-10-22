# Source code for iterative algorithm of projection-based tomography
# written and maintained by Zhen Guo
# --------------------------------------------------------

from torch.nn.functional import pad
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import time

def total_variation_loss(img, weight=1e-5):
    """
    TV regularization for an image
    """
    c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,1:,:]-img[:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,1:]-img[:,:,:-1], 2).sum()
    
    return weight*(tv_h+tv_w)/(c_img*h_img*w_img)

def Projection2D_batch(data3d, batch_size=100):
    """
    Genearting 2D projections from cone beam geometry with batching in the angles
    data3d in shape of nz, nx, ny
    """
    # to store multi-angle projections
    proj_tot = []
    
    data3d.requires_grad_()
    
    # batching for the angle
    for i in range(nProj // batch_size + 1):
        # genearting affine grids for sample rotation
        angle_rad = deg[i * batch_size:(i+1) * batch_size] / 360 * 2 * np.pi
        angle_rad = angle_rad.unsqueeze(-1).unsqueeze(-1)

        uu, vv = torch.meshgrid(us, vs)
        uu, vv = uu.T, vv.T
        xx, yy = torch.meshgrid(xs, ys)
        xx, yy = xx.T, yy.T

        rx = (xx * torch.cos(angle_rad) - yy * torch.sin(angle_rad)) / dx / nx * 2
        ry = (xx * torch.sin(angle_rad) + yy * torch.cos(angle_rad)) / dy / ny * 2

        if GPU:
            rx = rx.to(device='cuda:0')
            ry = ry.to(device='cuda:0')

        # rxry in (batch, rx, ry, 2)
        rxry = torch.stack((rx, ry), -1)

        # using bilinear interpolation to sample the rotated objects
        # expand the data3d in the batch dimension without consuming GPU memory
        # data3d_new in shape of (batch, nz, nx, ny)
        data3d_new = F.grid_sample(data3d.unsqueeze(0).expand(len(angle_rad), nz, nx, ny), rxry, 
                           mode='bilinear', padding_mode='zeros', align_corners=True)

        # generate sampling grid in xs and zs for cone beam
        xx, zz = torch.meshgrid(xs, zs)
        xx, zz = xx.T, zz.T

        Ratio = (ys + DSO) / DSD
        # to do broadcasting later on
        Ratio = Ratio.unsqueeze(-1).unsqueeze(-1)

        pu, pv = uu * Ratio, vv * Ratio
        pu, pv = (pu) / dx / nx * 2, (pv) / dz / nz * 2
        pu, pv = pu.to(torch.float32), pv.to(torch.float32)

        if GPU:
            pv = pv.to(device='cuda:0')
            pu = pu.to(device='cuda:0')

        # pupv in (ny, nu, nv, 2)
        pupv = torch.stack((pu, pv), -1)

        # permute data3d from (batch, nz, nx, ny) to (ny, batch, nz, nx)
        # to process all the ny elements
        data3d_new = data3d_new.permute(3, 0, 1, 2)
        temp = F.grid_sample(data3d_new, pupv, 
                             mode='bilinear', padding_mode='zeros', align_corners=True)

        # sum in ny to produce the 2d projections
        # then transpose to  (batch, nx, nz)
        proj2d = torch.sum(temp, dim=[0])
        proj2d = proj2d.permute(0, 2, 1)

        dist = torch.sqrt(DSD**2 + uu**2 + vv**2) / DSD * dy

        if GPU:
            dist = dist.to(device='cuda:0')
        # multiple the distance matrix
        proj2d = proj2d * dist.T
        
        proj_tot.append(proj2d)
    # concat alone the batch dimension to get all the projections
    proj_tot = torch.cat(proj_tot, dim=0)

    return proj_tot
  
def reconstruct(projections, resolution, lr, iterations, batch_size, 
                loss_func=torch.nn.MSELoss(), optimizer='Adam', GPU=False, schedule=True, TV_weight=0.5, clip=True):
    """Performs a full reconstruction based on gradient descent
    """
    
    # We start from a uniform image as our initial guess
    im = torch.zeros((resolution[0], resolution[1], resolution[2]))

    if GPU:
        im = im.to(device='cuda:0')
        projections = projections.to(device='cuda:0')

    if 'Adam'.lower() in optimizer.lower():
        t_optimizer = torch.optim.Adam([im], lr=lr)
    elif 'LBFGS'.lower() in optimizer.lower():
        t_optimizer = torch.optim.LBFGS([im], lr=lr, history_size=2, 
                                        tolerance_grad=1e-11, tolerance_change=1e-11, max_iter=10)
    elif 'sgd'.lower() in optimizer.lower():
        t_optimizer = torch.optim.SGD([im], lr=lr, momentum=True)

    if schedule:
        scheduler = torch.optim.lr_scheduler.StepLR(t_optimizer, step_size=20, gamma=0.5)

    def closure():
        t_optimizer.zero_grad()
        projected = Projection2D_batch(im, batch_size=batch_size)
        l1 = loss_func(projected, projections)
        l2 = total_variation_loss(im, weight=TV_weight)
        l = l1 + l2
        l.backward()
        return l

    loss = []
    results = []
    times = []
    for i in range(iterations):
        start = time.time()
        tensor = t_optimizer.step(closure)
        end = time.time()
        #print(end - start)
        tensor = tensor.detach().cpu().numpy()
        loss.append(tensor[()])
        times.append(end-start)
        
        if schedule:
            scheduler.step()
        
        if clip:
            with torch.no_grad():
                im.clamp_(min=0, max=1)
        
        result = im.detach()
        results.append(np.array(result.to(device='cpu')))

    return results, loss, times
