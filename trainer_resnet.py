import pandas as pd
import torch
from metric_tracker import MetricTracker
from network import resnet18
import os
import torch.utils.data as data
import torchvision.transforms as transforms
from evaluator import Evaluator
import numpy as np
from tqdm import tqdm
from h5_dataset import HDF5Dataset
import deep_taylor as sa_map
from torch.autograd import Variable
from losses import ClassDistinctivenessLoss, PrelogitsSaliencySimLoss


class ResnetTrainer():
    def __init__(self, opt):
        self.opt = opt

    def initialize(self):
        self.label_dict = {0: 'dwi',
                           1: 'flair',
                           2: 'swi',
                           3: 't1',
                           4: 't1c',
                           5: 't2'}

        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.Lambda(lambd=lambda x: (x + 1) / 2)
        ])
        if self.opt.test:
            self.train_loader = None
            self.val_loader = None
            # load the data
            self.test_dataset = HDF5Dataset(self.opt.test_data_path, self.data_transform)
            # encapsulate data into dataloader form
            self.test_loader = data.DataLoader(dataset=self.test_dataset, drop_last=False,
                                               batch_size=self.opt.batch_size, shuffle=False,
                                               num_workers=1)
            self.evaluator = Evaluator('test', self.opt.test_data_path, test_vol=self.opt.vol_test)

            print('Length of test dataset: {}'.format(self.test_dataset.__len__()))

        else:
            ### initialize dataloaders
            self.train_dataset = HDF5Dataset(self.opt.train_data_path, self.data_transform)
            self.val_dataset = HDF5Dataset(self.opt.val_data_path, self.data_transform)
            self.train_loader = data.DataLoader(dataset=self.train_dataset,
                                                batch_size=self.opt.batch_size,
                                                shuffle=True, drop_last=False,
                                                num_workers=self.opt.n_dataloader_workers)

            print('Length of train dataset: {}'.format(self.train_dataset.__len__()))
            self.val_loader = data.DataLoader(dataset=self.val_dataset,
                                              batch_size=self.opt.batch_size, shuffle=False,
                                              drop_last=False, num_workers=self.opt.n_dataloader_workers)
            print('Length of val dataset: {}'.format(self.val_dataset.__len__()))
            self.evaluator = Evaluator('val', self.opt.val_data_path)

        ## initialize the models
        self.model = resnet18(pretrained=False, num_classes=self.opt.num_classes)

        ## load models if needed
        if self.opt.continue_train:
            self.load_models(self.opt.resume_iter)
            print('-------- Load the model --------')

        ## use gpu
        if self.opt.use_gpu:
            self.model = self.model.to(self.opt.gpu_id)

        ## optimizers, schedulars
        self.optimizer = self.get_optimizers()
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

        ## losses
        self.ce = torch.nn.CrossEntropyLoss()
        self.class_distive = ClassDistinctivenessLoss(device=self.opt.gpu_id)
        self.pre_logit_sal_sim = PrelogitsSaliencySimLoss(device=self.opt.gpu_id)
        # self.min_sal_loss = ClassSaliencyMinLoss(device=self.opt.gpu_id)
        ## metrics
        self.metric_tracker = MetricTracker()

    def loss_function(self, logits, labels, saliency_list, pre_logits, alpha1 = 1, alpha2 = 1, alpha3 = 1):
        ce_loss = self.ce(logits, labels)
        class_distinctive_loss, class_distinctive_loss_list, class_distinctive_dis_list = self.class_distive(saliency_list)
        pre_logits_sim_loss = self.pre_logit_sal_sim(saliency_list,pre_logits,labels)
        # sal_min_loss = self.min_sal_loss(saliency_list)
        loss = ce_loss + alpha1*class_distinctive_loss + alpha2* pre_logits_sim_loss #+ alpha3*sal_min_loss
        return loss, ce_loss, class_distinctive_loss, pre_logits_sim_loss,class_distinctive_loss_list, class_distinctive_dis_list# sal_min_loss, class_distinctive_loss_list, class_distinctive_dis_list

    def load_models(self, epoch):
        checkpoints_dir = self.opt.checkpoints_dir

        weights = torch.load(os.path.join(checkpoints_dir, 'saved_models', 'Segmentor_%s.pth' % epoch),
                             map_location='cpu')
        self.model.load_state_dict(weights)

    def save_models(self, epoch):
        checkpoints_dir = self.opt.checkpoints_dir
        if not os.path.isdir(os.path.join(checkpoints_dir, 'saved_models')):
            os.makedirs(os.path.join(checkpoints_dir, 'saved_models'))
        torch.save(self.model.state_dict(), os.path.join(checkpoints_dir, 'saved_models', 'Segmentor_%s.pth' % epoch))

    def get_optimizers(self):
        params = list(self.model.parameters())
        optimizer = torch.optim.RMSprop(params, lr=self.opt.lr, weight_decay=1e-4, momentum=0.9)
        # optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # maximize dice score
        return optimizer#, scheduler

    ###################### training logic ################################
    def train_one_step(self, imgs, labels):
        # zero out previous grads
        self.optimizer.zero_grad()

        logits, pred_logits = self.model(imgs)
        sal_map_list = self.deep_taylor_in_train_step(imgs, labels)
        self.model.train()
        loss, ce_loss, class_distinctive_loss, pre_logits_sim_loss, class_distinctive_loss_sep, class_distinctive_dis_sep = self.loss_function(logits, labels, sal_map_list, pred_logits, alpha1 = self.opt.alpha_1, alpha2 = self.opt.alpha_2, alpha3 = self.opt.alpha_3)  # loss_ce + loss_dc
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        cla_losses = {}
        cla_losses['train_loss'] = loss.detach().cpu().numpy()
        cla_losses['train_ce_loss'] = ce_loss.detach().cpu().numpy()
        cla_losses['train_cd_loss_all'] = class_distinctive_loss.detach().cpu().numpy()
        cla_losses['train_pre_logit_loss'] = pre_logits_sim_loss.detach().cpu().numpy()
        for i in range(len(class_distinctive_loss_sep)):
            cla_losses[f'train_cd_loss_pair{i:02}'] = class_distinctive_loss_sep[i].detach().cpu().numpy()

        return cla_losses

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        y_true = torch.tensor([])
        y_score = torch.tensor([])
        cla_losses = {}

        data_loader = self.test_loader if self.opt.test else self.val_loader

        with torch.no_grad():
            loss_list = []
            ce_losses = []
            class_distinctive_loss_list = []
            pre_logits_sim_loss_list = []
            # sal_min_loss_list = []
            class_distinctive_loss_sep_list = []
            class_distinctive_dis_sep_list = []
            for i, (image, label) in enumerate(data_loader):
                if self.opt.use_gpu:
                    image = image.to(self.opt.gpu_id)
                    label = label.to(self.opt.gpu_id)
                    label = label.squeeze().long()
                logits, pre_logits = self.model(image)
                sal_map_list = self.deep_taylor_in_train_step(image, label)
                softmax = logits.softmax(dim=-1)
                if self.opt.output_pre_logits:
                    np.save(f'{self.opt.pre_logits_path}/pre_logits{i}.npy', pre_logits.detach().cpu().numpy())
                loss, ce_loss, class_distinctive_loss, pre_logits_sim_loss, class_distinctive_loss_sep, class_distinctive_dis_sep = self.loss_function(logits, label, sal_map_list, pre_logits, alpha1 = self.opt.alpha_1, alpha2 = self.opt.alpha_2, alpha3 = self.opt.alpha_3)
                loss_list.append(loss.detach().cpu().numpy())
                ce_losses.append(ce_loss.detach().cpu().numpy())
                class_distinctive_loss_list.append(class_distinctive_loss.detach().cpu().numpy())
                pre_logits_sim_loss_list.append(pre_logits_sim_loss.detach().cpu().numpy())
                # sal_min_loss_list.append(sal_min_loss.detach().cpu().numpy())
                class_distinctive_loss_sep_list.append(np.array([i.detach().cpu().numpy() for i in class_distinctive_loss_sep]))
                class_distinctive_dis_sep_list.append(np.array([i.detach().cpu().numpy() for i in class_distinctive_dis_sep]))
                label = label.to(torch.float32)

                y_score = torch.cat((y_score, softmax.cpu()), 0)
            y_score = y_score.detach().numpy()

            if self.opt.vol_test:
                y_score = np.mean(y_score.reshape(-1,7,6),axis = 1)
            else:
                metrics = self.evaluator.evaluate(y_score)
        cla_losses['eval_mean_loss'] = np.mean(loss_list)
        cla_losses['eval_std_loss'] = np.std(loss_list)
        cla_losses['eval_mean_ce'] = np.mean(ce_losses)
        cla_losses['eval_std_ce'] = np.std(ce_losses)
        cla_losses['eval_mean_cd_all'] = np.mean(class_distinctive_loss_list)
        cla_losses['eval_std_cd_all'] = np.std(class_distinctive_loss_list)
        cla_losses['eval_mean_pre_logit'] = np.mean(pre_logits_sim_loss_list)
        cla_losses['eval_std_pre_logit'] = np.std(pre_logits_sim_loss_list)
        class_distinctive_loss_sep_list_arr_means = np.mean(np.array(class_distinctive_loss_sep_list), axis = 0)
        class_distinctive_loss_sep_list_arr_stds = np.std(np.array(class_distinctive_loss_sep_list), axis = 0)
        class_distinctive_dis = np.concatenate(class_distinctive_dis_sep_list)

        for i in range(len(class_distinctive_loss_sep_list_arr_means)):
            cla_losses[f'eval_mean_cd_pair{i:02}'] = class_distinctive_loss_sep_list_arr_means[i]
            cla_losses[f'eval_std_cd_pair{i:02}'] = class_distinctive_loss_sep_list_arr_stds[i]
        if self.opt.test:
            print(f'Test on Epoch {self.opt.resume_iter} AUC: {metrics._asdict()["AUC"]}; ACC: {metrics._asdict()["ACC"]}; F1 {metrics._asdict()["F1"]}')
            print(self.opt.vol_test)
            if self.opt.vol_test:
                np.save(f'{self.opt.checkpoints_dir}/predict_probs_{self.opt.resume_iter}{self.opt.scalped}.npy', y_score)
            else:
                np.save(f'{self.opt.checkpoints_dir}/predict_probs_slice_{self.opt.resume_iter}{self.opt.scalped}.npy', y_score)
        else:
            print(f'Evaluate on current AUC: {metrics._asdict()["AUC"]}; ACC: {metrics._asdict()["ACC"]}; F1 {metrics._asdict()["F1"]}')
        self.model.train()

        return cla_losses, metrics._asdict(), class_distinctive_dis



    def train(self):
        epochs = self.opt.n_epochs
        epoch_record = {}
        for epoch in np.arange(epochs):
            epoch_record['epoch'] = epoch
            for image, label in tqdm(self.train_loader):

                if self.opt.use_gpu:
                    image = image.to(self.opt.gpu_id)
                    label = label.to(self.opt.gpu_id)
                    label = label.squeeze().long()
                losses = self.train_one_step(image, label)
            print('batch train loss at epoch {} is {}'.format(epoch, losses['train_loss']))
            if epoch % 5 == 0 or epoch % 2 == 0:
                self.metric_tracker.update_metrics(losses, smoothe=False)
            losses, metrics, sal_dis = self.eval()
            print('mean val losses at epoch {} is {}'.format(epoch, losses['eval_mean_loss']))
            if epoch % 5 == 0 or epoch % 2 == 0:
                self.metric_tracker.update_metrics(losses, smoothe=False)
                self.metric_tracker.update_metrics(metrics, smoothe=False)
                self.metric_tracker.update_metrics(epoch_record, smoothe=False)
            if epoch % 5 == 0 or epoch % 2 == 0:
                self.save_models(epoch)
                np.save(f'{self.opt.sal_dis_path}/sal_dis_{epoch}.npy',sal_dis)
                if epoch > 0:
                    df = pd.DataFrame(self.metric_tracker.metrics)
                    df.to_csv(f'{self.opt.checkpoints_dir}/eval_info.csv')

    def deep_taylor(self):
        model_archi = 'resnet'
        self.model.train(False)
        module_list = sa_map.model_flattening(self.model)
        act_store_model = sa_map.ActivationStoringNet(module_list)
        DTD = sa_map.DTD()
        if self.opt.use_gpu:
            act_store_model = act_store_model.to(self.opt.gpu_id)
            DTD = DTD.to(self.opt.gpu_id)
        sal_maps = []
        with torch.no_grad():
            for i, (image, label) in enumerate(self.test_loader):
                if self.opt.use_gpu:
                    image = image.to(self.opt.gpu_id)
                    label = label.to(self.opt.gpu_id)
                    label = label.squeeze().long()

                if self.opt.cal_all_sal:
                    Rs = [torch.eye(self.opt.num_classes)[[i] * image.shape[0]] for i in
                          range(self.opt.num_classes)]
                else:
                    Rs = [None]
                saliency_maps = []
                for R in Rs:
                    image = Variable(image)
                    label = Variable(label)
                    module_stack, output = act_store_model(image)
                    # print(len(module_stack))
                    saliency_maps.append(DTD(module_stack, output, self.opt.num_classes, model_archi, self.opt.gpu_id, R, layer_ind = self.opt.layer_ind))
                # saliency_maps = [torch.sum(saliency_map, dim=1).detach().cpu().numpy() for saliency_map in saliency_maps]
                saliency_maps = np.stack([saliency_map.detach().cpu().numpy() for saliency_map in
                                 saliency_maps], axis = -1)

                np.save(f'{self.opt.deep_taylor_savepath_layer}/sal_each_label_{i}.npy', saliency_maps)

    def deep_taylor_in_train_step(self, image, label):
        model_archi = 'resnet'
        self.model.eval()
        module_list = sa_map.model_flattening(self.model)

        act_store_model = sa_map.ActivationStoringNet(module_list)
        DTD = sa_map.DTD()
        if self.opt.use_gpu:
            act_store_model = act_store_model.to(self.opt.gpu_id)
            DTD = DTD.to(self.opt.gpu_id)
        Rs = [torch.eye(self.opt.num_classes)[[i] * image.shape[0]] for i in
              range(self.opt.num_classes)]
        saliency_maps = []
        for R in Rs:
            image = Variable(image)
            label = Variable(label)
            module_stack, output = act_store_model(image)
            saliency_maps.append(DTD(module_stack, output, self.opt.num_classes, model_archi, self.opt.gpu_id, R, layer_ind = self.opt.layer_ind))
        return saliency_maps

    def launch(self):
        self.initialize()
        if self.opt.test:
            if self.opt.deep_taylor:
                self.deep_taylor()
            else:
                self.eval()
        else:
            self.train()