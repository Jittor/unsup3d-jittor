import os
import glob
from datetime import datetime
import numpy as np
from . import meters
from . import utils
from .dataloaders import get_data_loaders
import jittor as jt

class Trainer():
    def __init__(self, cfgs, model):
        self.num_epochs = cfgs.get('num_epochs', 30)
        self.batch_size = cfgs.get('batch_size', 64)
        self.checkpoint_dir = cfgs.get('checkpoint_dir', 'results')
        self.checkpoint_load_dir = cfgs.get('checkpoint_load_dir', 'results')
        self.save_checkpoint_freq = cfgs.get('save_checkpoint_freq', 1)
        self.keep_num_checkpoint = cfgs.get('keep_num_checkpoint', 2)  # -1 for keeping all checkpoints
        self.resume = cfgs.get('resume', True)
        self.use_logger = cfgs.get('use_logger', False)
        self.log_freq = cfgs.get('log_freq', 1000)
        self.archive_code = cfgs.get('archive_code', True)
        self.checkpoint_name = cfgs.get('checkpoint_name', None)
        self.test_result_dir = cfgs.get('test_result_dir', None)
        self.cfgs = cfgs

        self.metrics_trace = meters.MetricsTrace()
        self.make_metrics = lambda m=None: meters.StandardMetrics(m)
        self.model = model(cfgs)
        self.model.trainer = self
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(cfgs)

    def load_checkpoint(self, optim=True):
        """Search the specified/latest checkpoint in checkpoint_dir and load the model and optimizer."""
        netD_path = os.path.join(self.checkpoint_load_dir, f'netD_last.pkl')
        netA_path = os.path.join(self.checkpoint_load_dir, f'netA_last.pkl')
        netL_path = os.path.join(self.checkpoint_load_dir, f'netL_last.pkl')
        netV_path = os.path.join(self.checkpoint_load_dir, f'netV_last.pkl')
        netC_path = os.path.join(self.checkpoint_load_dir, f'netC_last.pkl')

        self.model.netD.load(netD_path)
        self.model.netA.load(netA_path)
        self.model.netL.load(netL_path)
        self.model.netV.load(netV_path)
        self.model.netC.load(netC_path)

        print(f"Loading checkpoint from {self.checkpoint_load_dir}")

    def save_checkpoint(self, epoch, optim=True):
        """Save model, optimizer, and metrics state to a checkpoint in checkpoint_dir for the specified epoch."""
        utils.xmkdir(self.checkpoint_dir)
        netD_path = os.path.join(self.checkpoint_dir, f'netD_last.pkl')
        netA_path = os.path.join(self.checkpoint_dir, f'netA_last.pkl')
        netL_path = os.path.join(self.checkpoint_dir, f'netL_last.pkl')
        netV_path = os.path.join(self.checkpoint_dir, f'netV_last.pkl')
        netC_path = os.path.join(self.checkpoint_dir, f'netC_last.pkl')
        
        self.model.netD.save(netD_path)
        self.model.netA.save(netA_path)
        self.model.netL.save(netL_path)
        self.model.netV.save(netV_path)
        self.model.netC.save(netC_path)
        print(f"Saving checkpoint to {self.checkpoint_dir}")

    def test(self):
        """Perform testing."""
        self.current_epoch = self.load_checkpoint(optim=False)
        if self.test_result_dir is None:
            self.test_result_dir = os.path.join(self.checkpoint_dir, f'test_results_{self.checkpoint_name}'.replace('.pth',''))
        print(f"Saving testing results to {self.test_result_dir}")

        with jt.no_grad():
            m = self.run_epoch(self.test_loader, epoch=self.current_epoch, is_test=True)

        score_path = os.path.join(self.test_result_dir, 'eval_scores.txt')
        self.model.save_scores(score_path)

    def train(self):
        """Perform training."""
        ## archive code and configs
        if self.archive_code:
            utils.archive_code(os.path.join(self.checkpoint_dir, 'archived_code.zip'), filetypes=['.py', '.yml'])
        utils.dump_yaml(os.path.join(self.checkpoint_dir, 'configs.yml'), self.cfgs)

        ## initialize
        self.metrics_trace.reset()
        self.train_iter_per_epoch = len(self.train_loader)
        self.model.init_optimizers()

        ## resume from checkpoint
        if self.resume:
            self.load_checkpoint(optim=True)

        ## initialize tensorboardX logger
        if self.use_logger:
            from tensorboardX import SummaryWriter
            self.logger = SummaryWriter(os.path.join(self.checkpoint_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")))

            ## cache one batch for visualization
            self.viz_input = self.val_loader.__iter__().__next__()

        ## run epochs
        print(f"{self.model.model_name}: optimizing to {self.num_epochs} epochs")
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            metrics = self.run_epoch(self.train_loader, epoch)
            self.metrics_trace.append("train", metrics)

            with jt.no_grad():
                metrics = self.run_epoch(self.val_loader, epoch, is_validation=True)
                self.metrics_trace.append("val", metrics)

            if (epoch+1) % self.save_checkpoint_freq == 0:
                self.save_checkpoint(epoch+1, optim=True)   
            # self.metrics_trace.plot(pdf_path=os.path.join(self.checkpoint_dir, 'metrics.pdf'))
            # self.metrics_trace.save(os.path.join(self.checkpoint_dir, 'metrics.json'))

        print(f"Training completed after {epoch+1} epochs.")

    def run_epoch(self, loader, epoch=0, is_validation=False, is_test=False):
        """Run one epoch."""
        is_train = not is_validation and not is_test
        metrics = self.make_metrics()

        if is_train:
            print(f"Starting training epoch {epoch}")
            self.model.set_train()
        else:
            print(f"Starting validation epoch {epoch}")
            self.model.set_eval()

        for iter, input in enumerate(loader):
            if self.use_logger and is_train:
                total_iter = iter + epoch*self.train_iter_per_epoch
                if total_iter % self.log_freq == 0:
                    self.model.forward(self.viz_input)
                    self.model.visualize(self.logger, total_iter=total_iter, max_bs=25)

            m = self.model.forward(input)
            if is_train:
                self.model.backward()
            elif is_test:
                self.model.save_results(self.test_result_dir)

            metrics.update(m, self.batch_size)
            print(f"{'T' if is_train else 'V'}{epoch:02}/{iter:05}/{metrics}")

            
        return metrics
