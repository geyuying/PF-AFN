import time
from options.train_options import TrainOptions
from models.networks import ResUnetGenerator, VGGLoss, save_checkpoint, load_checkpoint_parallel
from models.afwm import TVLoss, AFWM
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
import cv2
import datetime

opt = TrainOptions().parse()
path = 'runs/'+opt.name
os.makedirs(path,exist_ok=True)

def CreateDataset(opt):
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

os.makedirs('sample',exist_ok=True)
opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

torch.cuda.set_device(opt.local_rank)
torch.distributed.init_process_group(
    'nccl',
    init_method='env://'
)
device = torch.device(f'cuda:{opt.local_rank}')

start_epoch, epoch_iter = 1, 0

train_data = CreateDataset(opt)
train_sampler = DistributedSampler(train_data)
train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False,
                                               num_workers=4, pin_memory=True, sampler=train_sampler)
dataset_size = len(train_loader)

warp_model = AFWM(opt, 45)
print(warp_model)
warp_model.train()
warp_model.cuda()
load_checkpoint_parallel(warp_model, opt.PBAFN_warp_checkpoint)

gen_model = ResUnetGenerator(8, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
print(gen_model)
gen_model.train()
gen_model.cuda()

warp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(warp_model).to(device)
gen_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gen_model).to(device)

if opt.isTrain and len(opt.gpu_ids):
    model = torch.nn.parallel.DistributedDataParallel(warp_model, device_ids=[opt.local_rank])
    model_gen = torch.nn.parallel.DistributedDataParallel(gen_model, device_ids=[opt.local_rank])

criterionL1 = nn.L1Loss()
criterionVGG = VGGLoss()
# optimizer
params_warp = [p for p in model.parameters()]
params_gen = [p for p in model_gen.parameters()]
optimizer_warp = torch.optim.Adam(params_warp, lr=0.2*opt.lr, betas=(opt.beta1, 0.999))
optimizer_gen = torch.optim.Adam(params_gen, lr=opt.lr, betas=(opt.beta1, 0.999))

total_steps = (start_epoch-1) * dataset_size + epoch_iter

step = 0
step_per_batch = dataset_size

if opt.local_rank == 0:
    writer = SummaryWriter(path)

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    train_sampler.set_epoch(epoch)

    for i, data in enumerate(train_loader):

        iter_start_time = time.time()

        total_steps += 1
        epoch_iter += 1
        save_fake = True

        t_mask = torch.FloatTensor((data['label'].cpu().numpy()==7).astype(np.float))
        data['label'] = data['label']*(1-t_mask)+t_mask*4
        edge = data['edge']
        pre_clothes_edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
        clothes = data['color']
        clothes = clothes * pre_clothes_edge
        person_clothes_edge = torch.FloatTensor((data['label'].cpu().numpy()==4).astype(np.int))
        real_image = data['image']
        person_clothes = real_image*person_clothes_edge
        pose = data['pose']
        size = data['label'].size()
        oneHot_size1 = (size[0], 25, size[2], size[3])
        densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1)).zero_()
        densepose = densepose.scatter_(1,data['densepose'].data.long().cuda(),1.0)
        densepose_fore = data['densepose']/24.0
        face_mask = torch.FloatTensor((data['label'].cpu().numpy()==1).astype(np.int))+torch.FloatTensor((data['label'].cpu().numpy()==12).astype(np.int))
        other_clothes_mask = torch.FloatTensor((data['label'].cpu().numpy()==5).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy()==6).astype(np.int))\
                             + torch.FloatTensor((data['label'].cpu().numpy()==8).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy()==9).astype(np.int))\
                             + torch.FloatTensor((data['label'].cpu().numpy()==10).astype(np.int))
        face_img = face_mask * real_image
        other_clothes_img = other_clothes_mask * real_image
        preserve_region = face_img + other_clothes_img
        preserve_mask = torch.cat([face_mask, other_clothes_mask],1)
        concat = torch.cat([preserve_mask.cuda(), densepose, pose.cuda()],1)
        arm_mask = torch.FloatTensor((data['label'].cpu().numpy()==11).astype(np.float)) + torch.FloatTensor((data['label'].cpu().numpy()==13).astype(np.float))
        hand_mask = torch.FloatTensor((data['densepose'].cpu().numpy()==3).astype(np.int)) + torch.FloatTensor((data['densepose'].cpu().numpy()==4).astype(np.int))
        hand_mask = arm_mask*hand_mask
        hand_img = hand_mask*real_image
        dense_preserve_mask = torch.FloatTensor((data['densepose'].cpu().numpy()==15).astype(np.int))+torch.FloatTensor((data['densepose'].cpu().numpy()==16).astype(np.int))\
                              +torch.FloatTensor((data['densepose'].cpu().numpy()==17).astype(np.int))+torch.FloatTensor((data['densepose'].cpu().numpy()==18).astype(np.int))\
                              +torch.FloatTensor((data['densepose'].cpu().numpy()==19).astype(np.int))+torch.FloatTensor((data['densepose'].cpu().numpy()==20).astype(np.int))\
                              +torch.FloatTensor((data['densepose'].cpu().numpy()==21).astype(np.int))+torch.FloatTensor((data['densepose'].cpu().numpy()==22))
        dense_preserve_mask = dense_preserve_mask.cuda()*(1-person_clothes_edge.cuda())
        preserve_region = face_img + other_clothes_img +hand_img

        flow_out = model(concat.cuda(), clothes.cuda(), pre_clothes_edge.cuda())
        warped_cloth, last_flow, _1, _2, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out

        epsilon = 0.001
        loss_smooth = sum([TVLoss(x) for x in delta_list])
        warp_loss = 0

        for num in range(5):
            cur_person_clothes = F.interpolate(person_clothes, scale_factor=0.5**(4-num), mode='bilinear')
            cur_person_clothes_edge = F.interpolate(person_clothes_edge, scale_factor=0.5**(4-num), mode='bilinear')
            loss_l1 = criterionL1(x_all[num], cur_person_clothes.cuda())
            loss_vgg = criterionVGG(x_all[num], cur_person_clothes.cuda())
            loss_edge = criterionL1(x_edge_all[num], cur_person_clothes_edge.cuda())
            b,c,h,w = delta_x_all[num].shape
            loss_flow_x = (delta_x_all[num].pow(2) + epsilon*epsilon).pow(0.45)
            loss_flow_x = torch.sum(loss_flow_x) / (b*c*h*w)
            loss_flow_y = (delta_y_all[num].pow(2) + epsilon*epsilon).pow(0.45)
            loss_flow_y = torch.sum(loss_flow_y) / (b*c*h*w)
            loss_second_smooth = loss_flow_x + loss_flow_y
            warp_loss = warp_loss + (num+1) * loss_l1 + (num+1) * 0.2 * loss_vgg + (num+1) * 2 * loss_edge + (num+1) * 6 * loss_second_smooth

        warp_loss = 0.01 * loss_smooth + warp_loss

        if opt.local_rank == 0:
          writer.add_scalar('warp_loss', warp_loss, step)

        warped_prod_edge = x_edge_all[4]
        gen_inputs = torch.cat([preserve_region.cuda(), warped_cloth, warped_prod_edge, dense_preserve_mask], 1)

        gen_outputs = model_gen(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite1 = m_composite * warped_prod_edge
        m_composite =  person_clothes_edge.cuda()*m_composite1
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        loss_mask_l1 = torch.mean(torch.abs(1 - m_composite))
        loss_l1 = criterionL1(p_tryon, real_image.cuda())
        loss_vgg = criterionVGG(p_tryon,real_image.cuda())
        bg_loss_l1 = criterionL1(p_rendered, real_image.cuda())
        bg_loss_vgg = criterionVGG(p_rendered, real_image.cuda())
        gen_loss = (loss_l1 * 5 + loss_vgg + bg_loss_l1 * 5 + bg_loss_vgg + loss_mask_l1)


        if opt.local_rank == 0:
          writer.add_scalar('gen_loss', gen_loss, step)

        loss_all = 0.5 * warp_loss + 1.0 * gen_loss

        if opt.local_rank == 0:
          writer.add_scalar('loss_all', loss_all, step)

        optimizer_warp.zero_grad()
        optimizer_gen.zero_grad()
        loss_all.backward()
        optimizer_warp.step()
        optimizer_gen.step()

        ############## Display results and errors ##########
        path = 'sample/'+opt.name
        os.makedirs(path,exist_ok=True)
        if step % 1000 == 0:
          if opt.local_rank == 0:
            a = real_image.float().cuda()
            b = person_clothes.cuda()
            c = clothes.cuda()
            d = torch.cat([densepose_fore.cuda(),densepose_fore.cuda(),densepose_fore.cuda()],1)
            e = warped_cloth
            f = torch.cat([warped_prod_edge,warped_prod_edge,warped_prod_edge],1)
            g = preserve_region.cuda()
            h = torch.cat([dense_preserve_mask,dense_preserve_mask,dense_preserve_mask],1)
            i = p_rendered
            j = torch.cat([m_composite1,m_composite1,m_composite1],1)
            k = p_tryon
            combine = torch.cat([a[0],b[0],c[0],d[0],e[0],f[0],g[0],h[0],i[0],j[0],k[0]], 2).squeeze()
            cv_img = (combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            writer.add_image('combine', (combine.data + 1) / 2.0, step)
            rgb = (cv_img*255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            cv2.imwrite('sample/'+opt.name+'/'+str(step)+'.jpg',bgr)

        step += 1
        iter_end_time = time.time()
        iter_delta_time = iter_end_time - iter_start_time
        step_delta = (step_per_batch-step%step_per_batch) + step_per_batch*(opt.niter + opt.niter_decay-epoch)
        eta = iter_delta_time*step_delta
        eta = str(datetime.timedelta(seconds=int(eta)))
        time_stamp = datetime.datetime.now()
        now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')

        if step % 100 == 0:
          if opt.local_rank == 0:
            print('{}:{}:[step-{}]--[loss-{:.6f}]--[loss-{:.6f}]--[ETA-{}]'.format(now, epoch_iter, step, warp_loss, gen_loss, eta))

        if epoch_iter >= dataset_size:
            break

    iter_end_time = time.time()
    if opt.local_rank == 0:
      print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
      if opt.local_rank == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        save_checkpoint(model.module, os.path.join(opt.checkpoints_dir, opt.name, 'PBAFN_warp_epoch_%03d.pth' % (epoch+1)))
        save_checkpoint(model_gen.module, os.path.join(opt.checkpoints_dir, opt.name, 'PBAFN_gen_epoch_%03d.pth' % (epoch+1)))

    if epoch > opt.niter:
        model.module.update_learning_rate_warp(optimizer_warp)
        model.module.update_learning_rate(optimizer_gen)
