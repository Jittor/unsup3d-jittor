import jittor as jt
from jittor import nn
from jittor import models

EPS = 1e-07

class Encoder(nn.Module):
    def __init__(self, cin, cout, nf=64, activation=nn.Tanh):
        super(Encoder, self).__init__()
        network = [
            nn.Conv(cin, nf, 4, stride=2, padding=1, bias=False), 
            nn.ReLU(), 
            nn.Conv(nf, (nf * 2), 4, stride=2, padding=1, bias=False), 
            nn.ReLU(), 
            nn.Conv((nf * 2), (nf * 4), 4, stride=2, padding=1, bias=False), 
            nn.ReLU(), 
            nn.Conv((nf * 4), (nf * 8), 4, stride=2, padding=1, bias=False), 
            nn.ReLU(), 
            nn.Conv((nf * 8), (nf * 8), 4, stride=1, padding=0, bias=False), 
            nn.ReLU(), 
            nn.Conv((nf * 8), cout, 1, stride=1, padding=0, bias=False)]
        if (activation is not None):
            network += [activation()]
        self.network = nn.Sequential(*network)

    def execute(self, input):
        return self.network(input).reshape([input.shape[0], (- 1)])

class EDDeconv(nn.Module):

    def __init__(self, cin, cout, zdim=128, nf=64, activation=nn.Tanh):
        super(EDDeconv, self).__init__()
        network = [
            nn.Conv(cin, nf, 4, stride=2, padding=1, bias=False), 
            nn.GroupNorm(16, nf), 
            nn.LeakyReLU(scale=0.2), 
            nn.Conv(nf, (nf * 2), 4, stride=2, padding=1, bias=False),
            nn.GroupNorm((16 * 2), (nf * 2)), 
            nn.LeakyReLU(scale=0.2), 
            nn.Conv((nf * 2), (nf * 4), 4, stride=2, padding=1, bias=False), 
            nn.GroupNorm((16 * 4), (nf * 4)), 
            nn.LeakyReLU(scale=0.2), 
            nn.Conv((nf * 4), (nf * 8), 4, stride=2, padding=1, bias=False), 
            nn.LeakyReLU(scale=0.2), 
            nn.Conv((nf * 8), zdim, 4, stride=1, padding=0, bias=False), 
            nn.ReLU()]
        network += [
            nn.ConvTranspose(zdim, (nf * 8), 4, stride=1, padding=0, bias=False), 
            nn.ReLU(), 
            nn.Conv((nf * 8), (nf * 8), 3, stride=1, padding=1, bias=False), 
            nn.ReLU(), 
            nn.ConvTranspose((nf * 8), (nf * 4), 4, stride=2, padding=1, bias=False), 
            nn.GroupNorm((16 * 4), (nf * 4)), 
            nn.ReLU(),
            nn.Conv((nf * 4), (nf * 4), 3, stride=1, padding=1, bias=False), 
            nn.GroupNorm((16 * 4), (nf * 4)), 
            nn.ReLU(), 
            nn.ConvTranspose((nf * 4), (nf * 2), 4, stride=2, padding=1, bias=False), 
            nn.GroupNorm((16 * 2), (nf * 2)), 
            nn.ReLU(), 
            nn.Conv((nf * 2), (nf * 2), 3, stride=1, padding=1, bias=False), 
            nn.GroupNorm((16 * 2), (nf * 2)), 
            nn.ReLU(), 
            nn.ConvTranspose((nf * 2), nf, 4, stride=2, padding=1, bias=False), 
            nn.GroupNorm(16, nf), nn.ReLU(), 
            nn.Conv(nf, nf, 3, stride=1, padding=1, bias=False), 
            nn.GroupNorm(16, nf), 
            nn.ReLU(), 
            nn.Upsample(scale_factor=2, mode='nearest'), 
            nn.Conv(nf, nf, 3, stride=1, padding=1, bias=False), 
            nn.GroupNorm(16, nf), nn.ReLU(), 
            nn.Conv(nf, nf, 5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16, nf), 
            nn.ReLU(), 
            nn.Conv(nf, cout, 5, stride=1, padding=2, bias=False)]
        if (activation is not None):
            network += [activation()]
        self.network = nn.Sequential(*network)

    def execute(self, input):
        return self.network(input)

class ConfNet(nn.Module):
    def __init__(self, cin, cout, zdim=128, nf=64):
        super(ConfNet, self).__init__()
        network = [
            nn.Conv(cin, nf, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, nf), 
            nn.LeakyReLU(scale=0.2), 
            nn.Conv(nf, (nf * 2), 4, stride=2, padding=1, bias=False), 
            nn.GroupNorm((16 * 2), (nf * 2)), 
            nn.LeakyReLU(scale=0.2), 
            nn.Conv((nf * 2), (nf * 4), 4, stride=2, padding=1, bias=False), 
            nn.GroupNorm((16 * 4), (nf * 4)), 
            nn.LeakyReLU(scale=0.2), 
            nn.Conv((nf * 4), (nf * 8), 4, stride=2, padding=1, bias=False), 
            nn.LeakyReLU(scale=0.2), 
            nn.Conv((nf * 8), zdim, 4, stride=1, padding=0, bias=False), 
            nn.ReLU()]
        network += [
            nn.ConvTranspose(zdim, (nf * 8), 4, padding=0, bias=False), 
            nn.ReLU(), nn.ConvTranspose((nf * 8), (nf * 4), 4, stride=2, padding=1, bias=False), 
            nn.GroupNorm((16 * 4), (nf * 4)), 
            nn.ReLU(), 
            nn.ConvTranspose((nf * 4), (nf * 2), 4, stride=2, padding=1, bias=False), 
            nn.GroupNorm((16 * 2), (nf * 2)), 
            nn.ReLU()]
        self.network = nn.Sequential(*network)
        out_net1 = [
            nn.ConvTranspose((nf * 2), nf, 4, stride=2, padding=1, bias=False), 
            nn.GroupNorm(16, nf), 
            nn.ReLU(), 
            nn.ConvTranspose(nf, nf, 4, stride=2, padding=1, bias=False), 
            nn.GroupNorm(16, nf), 
            nn.ReLU(), 
            nn.Conv(nf, 2, 5, stride=1, padding=2, bias=False), 
            nn.Softplus()]
        self.out_net1 = nn.Sequential(*out_net1)
        out_net2 = [
            nn.Conv((nf * 2), 2, 3, stride=1, padding=1, bias=False), 
            nn.Softplus()]
        self.out_net2 = nn.Sequential(*out_net2)

    def execute(self, input):
        out = self.network(input)
        return (self.out_net1(out), self.out_net2(out))

class PerceptualLoss(nn.Module):

    def __init__(self, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        self.mean_rgb = jt.array([0.485, 0.456, 0.406])
        self.std_rgb = jt.array([0.229, 0.224, 0.225])
        vgg_pretrained_features = models.vgg.vgg16().features
        vgg_pretrained_features.load('init_models/vgg_pretrained_features.pkl')
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.append(vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.append(vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.append(vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.append(vgg_pretrained_features[x])
        if (not requires_grad):
            for param in self.parameters():
                param = param.stop_grad()

    def normalize(self, x):
        out = ((x / 2) + 0.5)
        out = ((out - self.mean_rgb.view((1, 3, 1, 1))) / self.std_rgb.view((1, 3, 1, 1)))
        return out

    def __call__(self, im1, im2, mask=None, conf_sigma=None):
        im = jt.contrib.concat([im1, im2], dim=0)
        im = self.normalize(im)
        feats = []
        f = self.slice1(im)
        feats += [jt.chunk(f, 2, dim=0)]
        f = self.slice2(f)
        feats += [jt.chunk(f, 2, dim=0)]
        f = self.slice3(f)
        feats += [jt.chunk(f, 2, dim=0)]
        f = self.slice4(f)
        feats += [jt.chunk(f, 2, dim=0)]
        losses = []
        for (f1, f2) in feats[2:3]:
            loss = ((f1 - f2).sqr())
            if (conf_sigma is not None):
                loss = ((loss / ((2 * (conf_sigma.sqr())) + EPS)) + (conf_sigma + EPS).log())
            if (mask is not None):
                (b, c, h, w) = loss.shape
                (_, _, hm, wm) = mask.shape
                (sh, sw) = ((hm // h), (wm // w))
                assert sh == sw
                mask0 = nn.Pool(kernel_size=sh, stride=sw, op="mean")(mask).broadcast(loss)
                loss = ((loss * mask0).sum() / mask0.sum())
            else:
                loss = loss.mean()
            losses += [loss]
        return sum(losses)
