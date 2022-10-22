from torch.nn.functional import pad
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

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

def total_variation_loss(img, weight=1e-5):
    """total variation loss for images
    """
    c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,1:,:]-img[:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,1:]-img[:,:,:-1], 2).sum()
    return weight*(tv_h+tv_w)/(c_img*h_img*w_img)

def reconstruct(projections, resolution, lr, iterations, batch_size, 
                loss_func=torch.nn.MSELoss(), optimizer='Adam', GPU=False, schedule=True, weight=0.5, clip=True):
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
        scheduler = torch.optim.lr_scheduler.StepLR(t_optimizer, step_size=10, gamma=0.5)

    def closure():
        t_optimizer.zero_grad()
        projected = Projection2D_batch(im, batch_size=batch_size)
        l1 = loss_func(projected, projections)
        l2 = total_variation_loss(im, weight=weight)
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

if __name__ == "__main__":
    
    import sys
    
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
    
    if my_task_id <= 11:
    
        the_idx = my_task_id-1

        direction = 1 # clock wise/ counter clockwise
        start = 0.
        end = 360.
        step = 360/1601
        deg = direction * torch.arange(start, end-step, step)
        nProj = len(deg)

        print(nProj)

        for TV_weight in [0, 2]:

            # non-negativity constraint?
            clip = True

            # using GPU?
            GPU = True

            # define number of object voxels
            nx, ny, nz = 300//2, 300//2, 1

            # object real size in mm
            sx, sy, sz = 8, 8, 16/128

            # real detector pixel density
            nu, nv = 128*2, 1

            # detector real size in mm
            su, sv = 25, 25/256

            # single voxel size
            dx, dy, dz, du, dv = sx/nx, sy/ny, sz/nz, su/nu, sv/nv

            # Geometry calculation
            xs = torch.arange(-(nx -1) / 2, nx / 2, 1) * dx
            ys = torch.arange(-(ny -1) / 2, ny / 2, 1) * dy
            zs = torch.arange(-(nz -1) / 2, nz / 2, 1) * dz

            us = torch.arange(-(nu -1) / 2, nu / 2, 1) * du
            vs = torch.arange(-(nv -1) / 2, nv / 2, 1) * dv

            # source to detector, and source to object axis distances in mm
            DSD, DSO = 230.11+342.17, 230.11

            tot_projections = np.load('tot_proj_ic_1600.npy')[1000*the_idx:1000*(the_idx+1)]

            total_recon = []

            for tot_projection in tqdm(tot_projections):

                recon, loss, times = reconstruct(torch.from_numpy(tot_projection[:, :, :]).to(torch.float32), 
                                                 [nz, nx, ny], lr=0.01, iterations=51, batch_size=1600, 
                                                 loss_func=torch.nn.MSELoss(), optimizer='Adam', 
                                                 GPU=True, schedule=True, weight=TV_weight, clip=clip)

                recon = recon[::10]

                total_recon.append(recon)

            total_recon = np.array(total_recon).astype(np.float32)
            print(np.shape(total_recon))
            np.save('full_angle_ic_1600_recons_' + str(the_idx) + '_weight_' + str(TV_weight) + '_clip_' + str(clip) + '.npy', total_recon)
            
    elif my_task_id > 11 and my_task_id <=30:
        
        photons = [32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400, 500, 640, 800, 1000, 1280, 1600, 2000]
        photon = photons[my_task_id-11-1]
        
        for offset in [40, 50]:
            
            for TV_weight in [0, 2]:
                
                for clip in [True, False]:

                    direction = 1 # clock wise/ counter clockwise
                    start = 0.
                    end = 360.
                    step = 360/1601 * offset
                    deg = direction * torch.arange(start, end-step, step)
                    nProj = len(deg)

                    print(nProj)

                    # using GPU?
                    GPU = True

                    # define number of object voxels
                    nx, ny, nz = 300//2, 300//2, 1

                    # object real size in mm
                    sx, sy, sz = 8, 8, 16/128

                    # real detector pixel density
                    nu, nv = 128*2, 1

                    # detector real size in mm
                    su, sv = 25, 25/256

                    # single voxel size
                    dx, dy, dz, du, dv = sx/nx, sy/ny, sz/nz, su/nu, sv/nv

                    # Geometry calculation
                    xs = torch.arange(-(nx -1) / 2, nx / 2, 1) * dx
                    ys = torch.arange(-(ny -1) / 2, ny / 2, 1) * dy
                    zs = torch.arange(-(nz -1) / 2, nz / 2, 1) * dz

                    us = torch.arange(-(nu -1) / 2, nu / 2, 1) * du
                    vs = torch.arange(-(nv -1) / 2, nv / 2, 1) * dv

                    # source to detector, and source to object axis distances in mm
                    DSD, DSO = 230.11+342.17, 230.11

                    sparse_recons = []

                    sparse_proj = np.load('tot_proj_test_ic_1600_photon_' + str(photon) + '.npy')

                    for sparse_proj in tqdm(sparse_proj):

                        recon, loss, times = reconstruct(torch.from_numpy(sparse_proj[::offset, :, :]).to(torch.float32), 
                                                         [nz, nx, ny], lr=0.01, iterations=51, batch_size=1600,
                                                         loss_func=torch.nn.MSELoss(), optimizer='Adam', GPU=True, 
                                                         schedule=True, weight=TV_weight, clip=clip)
                        recon = recon[::5]
                        sparse_recons.append(recon)

                    sparse_recons = np.array(sparse_recons).astype(np.float32)
                    print(np.shape(sparse_recons))
                    np.save('./iterative_results/iterative_sparse_photon_' + str(photon) 
                            + '_offset_' + str(offset) + '_weight_' + str(TV_weight) + '_clip_' + str(clip) + '.npy', sparse_recons)
                    
    elif my_task_id >30:
        
        the_idx = my_task_id-1-30
        
        for offset in [40, 50]:
            
            for TV_weight in [0, 2]:
                
                for clip in [True, False]:

                    direction = 1 # clock wise/ counter clockwise
                    start = 0.
                    end = 360.
                    step = 360/1601 * offset
                    deg = direction * torch.arange(start, end-step, step)
                    nProj = len(deg)

                    print(nProj)

                    # using GPU?
                    GPU = True

                    # define number of object voxels
                    nx, ny, nz = 300//2, 300//2, 1

                    # object real size in mm
                    sx, sy, sz = 8, 8, 16/128

                    # real detector pixel density
                    nu, nv = 128*2, 1

                    # detector real size in mm
                    su, sv = 25, 25/256

                    # single voxel size
                    dx, dy, dz, du, dv = sx/nx, sy/ny, sz/nz, su/nu, sv/nv

                    # Geometry calculation
                    xs = torch.arange(-(nx -1) / 2, nx / 2, 1) * dx
                    ys = torch.arange(-(ny -1) / 2, ny / 2, 1) * dy
                    zs = torch.arange(-(nz -1) / 2, nz / 2, 1) * dz

                    us = torch.arange(-(nu -1) / 2, nu / 2, 1) * du
                    vs = torch.arange(-(nv -1) / 2, nv / 2, 1) * dv

                    # source to detector, and source to object axis distances in mm
                    DSD, DSO = 230.11+342.17, 230.11

                    sparse_recons = []

                    sparse_proj = np.load('tot_proj_ic_1600.npy')[1000*the_idx:1000*(the_idx+1)]

                    for sparse_proj in tqdm(sparse_proj):

                        recon, loss, times = reconstruct(torch.from_numpy(sparse_proj[::offset, :, :]).to(torch.float32), 
                                                         [nz, nx, ny], lr=0.01, iterations=51, batch_size=1600,
                                                         loss_func=torch.nn.MSELoss(), optimizer='Adam', GPU=True, 
                                                         schedule=True, weight=TV_weight, clip=clip)
                        recon = recon[::5]
                        sparse_recons.append(recon)

                    sparse_recons = np.array(sparse_recons).astype(np.float32)
                    print(np.shape(sparse_recons))
                    np.save('./iterative_results/iterative_sparse_' + str(the_idx) 
                            + '_offset_' + str(offset) + '_weight_' + str(TV_weight) + '_clip_' + str(clip) + '.npy', sparse_recons)