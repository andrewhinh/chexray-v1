from IPython.display import display, HTML, clear_output
import torch
device = torch.device("cpu")
from fastai.vision.all import *
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
import warnings
warnings.filterwarnings('ignore')
from torch.distributions.beta import Beta
from fastai.text.all import *
from fastai.tabular.all import *
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence 
from torch.nn import functional as F, init
from torchvision import transforms, models
import pickle
from functools import partial
import copy as cp
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
import matplotlib.cm as cm
import skimage.transform
from fastai.data.load import _FakeLoader, _loaders

#Img Cap
imcap_path = Path('./files/imgcap/')
prod_path = Path('./files/imgcap/')
with open(imcap_path/'vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

class Fake(object):
    def __init__(self, li_obj):
        self.obj = li_obj
def add_text_to_num(df, vocab):
    df.insert(len(df.columns), 'numpath', np.nan, True)
    for row in range(len(df)):
        print(str(((row+1)/len(df))*100), "% done")
        clear_output(wait=True)
        temp = []
        report = ['<start>']+df.loc[row, 'text']+['<end>']
        for word in report:
            temp.append(vocab.index(word))
        df.at[row, 'numpath'] = Fake(temp)
    df.numpath = df.numpath.apply(lambda x: x.obj)

class ImageCaptionDataset(Dataset):
    def __init__(self, data, view='images', split=False, transform=None):
        data = data[data.split == split]
        self.filenames = list(data[view])
        self.captions  = list(data['numpath'])
        self.inds = data.index.tolist()
        self.transform = transform
    def __len__(self):
        return len(self.filenames)   
    def __getitem__(self, idx):
        image = PILImageBW.create(self.filenames[idx])
        caption = self.captions[idx]
        ref_ind = self.inds[idx]
        if self.transform is not None:
            image = self.transform(image)
        return (image, caption, ref_ind)

class TestImageCaptionDataset(ImageCaptionDataset):
    def __init__(self, data, view='images', transform=None):
        self.filenames = list(data[view])
        self.captions  = list(data['numpath'])
        self.inds = data.index.tolist()
        self.transform = transform

def pad_collate_ImgCap(samples, pad_idx = 0, pad_first:bool=True, backwards:bool=False, transpose:bool=False, device = device):
    "Function that collect samples and adds padding. Flips token order if needed"
    images, captions, ref_ind = zip(*samples)
    max_len_cap = max([len(c) for c in captions])
    decode_lengths = torch.tensor([len(c) for c in captions])
    res_cap = torch.zeros(len(samples), max_len_cap).long() + pad_idx
    ref_ind = torch.tensor(ref_ind)
    
    if backwards: pad_first = not pad_first
    for i,c in enumerate(captions):
        if pad_first: 
            res_cap[i,-len(c):] = LongTensor(c)
        else:         
            res_cap[i,:len(c)] = LongTensor(c)
    
    if backwards:
        cap = cap.flip(1)
    if transpose:
        res_cap.transpose_(0,1)
    
    images = torch.stack(images, 0, out=None)
    
    # Sort input data by decreasing lengths; why? apparent below
    decode_lengths, sort_ind = decode_lengths.sort(dim=0, descending=True)   
    images = images[sort_ind].to(device)
    res_cap = res_cap[sort_ind].to(device)
    ref_ind = ref_ind[sort_ind].to(device)
    decode_lengths = decode_lengths.to(device)
    
    return (images, res_cap, decode_lengths,ref_ind), res_cap[:, 1:]

class Encoder(nn.Module):
    def __init__(self, encode_img_size, fine_tune = True):
        super(Encoder, self).__init__()
        self.enc_imgsize = encode_img_size
        resnet = models.resnet34(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2]) # removing final Linear layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encode_img_size,encode_img_size))
        self.fine_tune = fine_tune
        self.fine_tune_h()
        
    def fine_tune_h(self):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.encoder.children())[5:]:
            for p in c.parameters():
                p.requires_grad = self.fine_tune
        
    def forward(self,X):
        out = self.encoder(X) # X is tensor of size (batch size, 3 (RGB), input height, width)
        out = self.adaptive_pool(out) # output (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)
        out = out.view(out.size(0), -1, out.size(3))
        return out
    
class Decoder(nn.Module):
    def __init__(self,attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5, pretrained_embedding = None,teacher_forcing_ratio = 0):
        super(Decoder, self).__init__()
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.vocab_size = vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim) 
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True) #use 
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # gate
        self.pretrained_embedding = pretrained_embedding
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()
        
    def init_weights(self):
        """
        Initilizes some parametes with values from the uniform Dist

        """
        self.embedding.weight.data.uniform_(0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1,0.1)

    def pretrained(self):
        if self.pretrained_embedding is not None:
            self.embedding.weight.data = torch.from_numpy(self.pretrained_embedding)
            
    def init_hidden_state(self, encoder_out):
        
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c
            
    def forward(self,encoder_out,encoded_captions,decode_lengths,inds):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        num_pixels = encoder_out.size(1)
        
        ## initililize hidden encoding
        h, c = self.init_hidden_state(encoder_out)
        
        decode_lengths = decode_lengths - 1
        
        max_len = max(decode_lengths).item()
         
        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max_len, vocab_size)
        alphas = torch.zeros(batch_size, max_len, num_pixels)
        
        for t in range(max_len):
            batch_size_t = sum([l.item() > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            # teacher forcing 
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
            
            inp_emb = self.embedding(encoded_captions[:batch_size_t,t]).float() if  (use_teacher_forcing or t==0) else self.embedding(prev_word[:batch_size_t]).float()
            #self.emb2dec_dim((embeddings[:batch_size_t, t, :]).float()) use syntax for teacher forcing
            #inp_emb = inp_emb if (use_teacher_forcing or t==0) else dec_out.squeeze(0)[:batch_size_t] #uncomment to add teacher forcing
            
            h, c = self.lstm(
                torch.cat([inp_emb, attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t,t , :] = preds
            alphas[:batch_size_t, t, :] = alpha

            _,prev_word = preds.max(dim=-1)
        return predictions, decode_lengths, alphas, inds
        
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        
        self.enc_att = nn.Linear(encoder_dim,attention_dim)
        self.dec_att = nn.Linear(decoder_dim,attention_dim)
        self.att = nn.Linear(attention_dim,1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,encoder_out, decoder_hidden):
        encoder_att = self.enc_att(encoder_out)
        decoder_att = self.dec_att(decoder_hidden)
        att = self.att(self.relu(encoder_att + decoder_att.unsqueeze(1))).squeeze(2) #testing added batchnorm 
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out*alpha.unsqueeze(2)).sum(dim=1)
        
        return attention_weighted_encoding, alpha

# Models Ensemble 
class Ensemble(nn.Module):
    def __init__(self,encoder, decoder):
        super(Ensemble, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self,x1): # you need flatten in between,caps,decode_lengths,inds
        imgs = self.encoder(torch.cat((torch.as_tensor(x1[0]),)*3, axis=1)) # here input x1 is Images: output (batch_size, encoded_image_size, encoded_image_size, 2048
        scores,decode_lengths,alphas,inds = self.decoder(imgs, x1[1], x1[2], x1[3]) #caps_sorted, decode_lengths, alphas, sort_ind
        return scores,decode_lengths,alphas,inds

class MixedDL():
    def __init__(self, pa_dl, lat_dl, device=device):
        "Stores away `tab_dl` and `vis_dl`, and overrides `shuffle_fn`"
        self.device = device
        pa_dl.shuffle_fn = self.shuffle_fn
        lat_dl.shuffle_fn = self.shuffle_fn
        self.dls = [pa_dl, lat_dl]
        self.count = 0
        self.fake_l = _FakeLoader(self, False, 0, 0)
    
    def __len__(self): return len(self.dls[0])
        
    def shuffle_fn(self, idxs):
        "Generates a new `rng` based upon which `DataLoader` is called"
        if self.count == 0: # if we haven't generated an rng yet
            self.rng = self.dls[0].rng.sample(idxs, len(idxs))
            self.count += 1
            return self.rng
        else:
            self.count = 0
            return self.rng
        
    def to(self, device): self.device = device

@patch
def __iter__(dl:MixedDL):
    "Iterate over your `DataLoader`"
    z = zip(*[_loaders[i.fake_l.num_workers==0](i.fake_l) for i in dl.dls])
    for b in z:
        if dl.device is not None: 
            b = to_device(b, dl.device)
        batch = []
        batch.append(dl.dls[0].after_batch(b[0][0]))
        batch.append(dl.dls[1].after_batch(b[1][0]))
        try: # In case the data is unlabelled
            batch.append(b[1][1])
            yield tuple(batch)
        except:
            yield tuple(batch)
@patch
def one_batch(x:MixedDL):
    "Grab a batch from the `DataLoader`"
    with x.fake_l.no_multiproc(): res = first(x)
    if hasattr(x, 'it'): delattr(x, 'it')
    return res

def get_pa_logits(self, inp, out):
    global glb_pa_logits
    glb_pa_logits = inp
    #return None

def get_lat_logits(self, inp, out):
    global glb_lat_logits
    glb_lat_logits = inp
    #return None

emb_dim = 300   # 300: pretrined words embedd GLove 
attention_dim = 512 # encoder_dim tranformed to attention_dim
decoder_dim = 512  #  word_emb_dim tranformed to decoder_dim
dropout = 0.5
encoder_dim = 512 #512 for resnet34 and 2048 for resnet 101 
vocab_size = len(vocab)

embExport_pkl_path = imcap_path/'Fastext_embedd_wordMap.pkl'
with open(embExport_pkl_path,'rb') as f:
    embedding = pickle.load(f)
    embedding[4021:] = np.random.normal(embedding.mean(),embedding.std(),(embedding.shape[0]-4021,300))

class MultViewCap(nn.Module):
    def __init__(self, pa_model, lat_model, num_classes=2): 
        super(MultViewCap, self).__init__()
        self.pa_model = pa_model
        self.lat_model = lat_model
        
        self.mixed_cls = nn.Linear(decoder_dim*2, vocab_size)
        self.pa_cls = nn.Linear(decoder_dim, vocab_size)
        self.lat_cls = nn.Linear(decoder_dim, vocab_size)
        
        #self.print_handle = self.tab_model.layers[2][0].register_forward_hook(printnorm)
        self.pa_handle = list(list(self.pa_model.children())[1].children())[-1].register_forward_hook(get_pa_logits)
        self.lat_handle = list(list(self.lat_model.children())[1].children())[-1].register_forward_hook(get_lat_logits)
    
    def remove_my_hooks(self):
        self.pa_handle.remove()
        self.lat_handle.remove()
        #self.print_handle.remove()
        return None
        
    def forward(self, x_pa, x_lat): 
        # PA Classifier
        pa_pred, decode_lengths, pa_alphas, _ = self.pa_model(x_pa)        
        
        # Lat Classifier
        lat_pred, _, lat_alphas, _ = self.lat_model(x_lat)
        
        # Logits
        pa_logits = glb_pa_logits[0]   # Only grabbling weights, not bias'
        lat_logits = glb_lat_logits[0]   # Only grabbling weights, not bias'
        mixed = torch.cat((pa_logits, lat_logits), dim=1)
        
        alphas = pa_alphas.add(lat_alphas) 
        alphas = alphas/2 

        # Mixed Classifier
        mixed_pred = self.mixed_cls(mixed)
        batch_size = 25 #Remember to change when bs changes
        max_len = max(decode_lengths).item() 
        predictions = torch.zeros(batch_size, max_len, vocab_size)
        for t in range(max_len):
            batch_size_t = sum([l.item() > t for l in decode_lengths])
            if not t:
                if batch_size_t<batch_size:
                    predictions = torch.zeros(batch_size_t, max_len, vocab_size)
            if list(mixed_pred.shape)[0]>1:
                temp = torch.zeros([vocab_size])
                for i in range(0, list(mixed_pred.shape)[0]):
                    temp=torch.add(temp, mixed_pred[i])
                mixed_pred=torch.div(temp, list(mixed_pred.shape)[0])
                mixed_pred=mixed_pred.unsqueeze(0)
            pred = torch.cat((mixed_pred,)*batch_size_t, axis=0)
            predictions[:batch_size_t,t , :] = pred
            
        return (pa_pred, lat_pred, predictions, decode_lengths, alphas)

def ModCELoss(pred, targets, decode_lengths, alphas, lamb=1, ce=True):
    pred = pack_padded_sequence(pred, decode_lengths, batch_first=True).to(device)
    targs = pack_padded_sequence(targets, decode_lengths, batch_first=True).to(device)
    if ce:
        loss = nn.CrossEntropyLoss().to(device)(pred.data, targs.data.long())
    else:
        loss = nn.CrossEntropyLossWithLogits().to(device)(pred.data, targs.data.long())
    loss += (lamb*((1. - alphas.sum(dim=1)) ** 2.).mean()).to(device) #stochastic attention
    return loss

class myGradientBlending(nn.Module):
    def __init__(self, pa_weight=0.0, lat_weight=0.0, pa_lat_weight=1.0, loss_scale=1.0, use_cel=True):
        "Expects weights for each model, the combined model, and an overall scale"
        super(myGradientBlending, self).__init__()
        self.pa_weight = pa_weight
        self.lat_weight = lat_weight
        self.pa_lat_weight = pa_lat_weight
        self.ce =  use_cel
        self.scale = loss_scale
        
    def forward(self, xb, yb):
        pa_out, lat_out, pl_out, decode_lengths, alphas = xb
        targ = yb
        "Gathers `self.loss` for each model, weighs, then sums"
        pa_loss = ModCELoss(pa_out, targ, decode_lengths, alphas, self.ce) * self.scale
        lat_loss = ModCELoss(lat_out, targ, decode_lengths, alphas, self.ce) * self.scale
        pl_loss = ModCELoss(pl_out, targ, decode_lengths, alphas, self.ce) * self.scale
        
        weighted_pa_loss = pa_loss * self.pa_weight
        weighted_lat_loss = lat_loss * self.lat_weight
        weighted_pl_loss = pl_loss * self.pa_lat_weight
        
        loss = weighted_pa_loss + weighted_lat_loss + weighted_pl_loss
        return loss

class TeacherForcingCallbackAll(Callback):
    def __init__(self, learn:Learner):
        super().__init__()
        self.learn = learn
    
    def on_batch_begin(self, epoch,**kwargs):
        self.learn.model.pa_model.decoder.teacher_forcing_ratio = (10 - epoch) * 0.1 if epoch < 10 else 0
        self.learn.model.lat_model.decoder.teacher_forcing_ratio = (10 - epoch) * 0.1 if epoch < 10 else 0
        
    def on_batch_end(self,**kwargs):
        self.learn.model.pa_model.decoder.teacher_forcing_ratio = 0.
        self.learn.model.lat_model.decoder.teacher_forcing_ratio = 0.
           
class CutMixImgCapAll(Callback):
    "Implementation of `https://arxiv.org/abs/1905.04899`"
    run_after,run_valid = [Normalize],False
    def __init__(self, alpha=1.): self.distrib = Beta(tensor(alpha), tensor(alpha))
        
    def before_fit(self):
        self.stack_y = getattr(self.learn.loss_func, 'y_int', False)
        if self.stack_y: self.old_lf,self.learn.loss_func = self.learn.loss_func,self.lf

    def after_fit(self):
        if self.stack_y: self.learn.loss_func = self.old_lf
            
    def before_batch(self):
        W, H = self.xb[0][0].size(3), self.xb[0][0].size(2)
        lam = self.distrib.sample((1,)).squeeze().to(self.x[0][0].device)
        lam = torch.stack([lam, 1-lam])
        self.lam = lam.max()
        shuffle = torch.randperm(self.y.size(0)).to(self.x[0][0].device)
        xb1,self.yb1 = tuple(L(self.xb[0]).itemgot(shuffle)),tuple(L(self.yb).itemgot(shuffle))
        nx_dims = len(self.x[0][0].size())
        
        W1, H1 = self.xb[1][0].size(3), self.xb[1][0].size(2)
        lam1 = self.distrib.sample((1,)).squeeze().to(self.x[1][0].device)
        lam1 = torch.stack([lam1, 1-lam1])
        self.lam1 = lam1.max()
        shuffle1 = torch.randperm(self.y.size(0)).to(self.x[1][0].device)
        xb1a,self.yb1a = tuple(L(self.xb[1]).itemgot(shuffle1)),tuple(L(self.yb).itemgot(shuffle1))
        nx_dims1 = len(self.x[1][0].size())
        
        x1, y1, x2, y2, x1a, y1a, x2a, y2a = self.rand_bbox(W, H, self.lam, W1, H1, self.lam1)
        self.learn.xb[0][0][:, :, x1:x2, y1:y2] = xb1[0][:, :, x1:x2, y1:y2]
        self.lam = (1 - ((x2-x1)*(y2-y1))/float(W*H)).item()
        
        self.learn.xb[1][0][:, :, x1a:x2a, y1a:y2a] = xb1a[0][:, :, x1:x2, y1:y2]
        self.lam1 = (1 - ((x2a-x1a)*(y2a-y1a))/float(W1*H1)).item()
    
    def lf(self, pred, *yb):
        if not self.training: return self.old_lf(pred, *yb)
        with NoneReduce(self.old_lf) as lf:
            loss = torch.lerp(lf(pred,*self.yb1), lf(pred,*yb), self.lam)
            loss1 = torch.lerp(lf(pred,*self.yb1a), lf(pred,*yb), self.lam1)
        return reduce_loss((loss+loss1)/2, getattr(self.old_lf, 'reduction', 'mean'))

    def rand_bbox(self, W, H, lam, W1, H1, lam1):
        cut_rat = torch.sqrt(1. - lam)
        cut_w = (W * cut_rat).type(torch.long)
        cut_h = (H * cut_rat).type(torch.long)
        # uniform
        cx = torch.randint(0, W, (1,)).to(self.x[0][0].device)
        cy = torch.randint(0, H, (1,)).to(self.x[0][0].device)
        x1 = torch.clamp(cx - cut_w // 2, 0, W)
        y1 = torch.clamp(cy - cut_h // 2, 0, H)
        x2 = torch.clamp(cx + cut_w // 2, 0, W)
        y2 = torch.clamp(cy + cut_h // 2, 0, H)
        
        cut_rat1 = torch.sqrt(1. - lam1)
        cut_w1 = (W1 * cut_rat1).type(torch.long)
        cut_h1 = (H1 * cut_rat1).type(torch.long)
        # uniform
        cx1 = torch.randint(0, W1, (1,)).to(self.x[1][0].device)
        cy1 = torch.randint(0, H1, (1,)).to(self.x[1][0].device)
        x1a = torch.clamp(cx1 - cut_w1 // 2, 0, W1)
        y1a = torch.clamp(cy1 - cut_h1 // 2, 0, H1)
        x2a = torch.clamp(cx1 + cut_w1 // 2, 0, W1)
        y2a = torch.clamp(cy1 + cut_h1 // 2, 0, H1)
        return x1, y1, x2, y2, x1a, y1a, x2a, y2a

# Acc Metrics
def compute_acc(pred, decode_lengths, targets, k):
    batch_size = targets.size(0)
    scores = pack_padded_sequence(pred, decode_lengths, batch_first=True).to(device)
    targ = pack_padded_sequence(targets, decode_lengths, batch_first=True).to(device)
    batch_size = targ.data.size(0)
    _, ind = scores.data.topk(k, 1, True, True)
    correct = ind.eq(targ.data.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total * (100.0 / batch_size)

def topK_accuracy_pa(input, targets, k=5):
    pred, _, _, decode_lengths, _ = input
    return compute_acc(pred, decode_lengths, targets, k)

def topK_accuracy_lat(input, targets, k=5):
    _, pred, _, decode_lengths, _ = input
    return compute_acc(pred, decode_lengths, targets, k)

def topK_accuracy_pl(input, targets, k=5):
    _, _, pred, decode_lengths, _ = input
    return compute_acc(pred, decode_lengths, targets, k)

def weighted_accuracy(inp, targ, axis=-1, w_pa=0.333, w_lat=0.333, w_pl=0.333, k=5):
    pa_pred, lat_pred, pl_pred, decode_lengths, _ = inp
    pa_inp = pa_pred * w_pa
    lat_inp = lat_pred * w_lat
    pl_inp = pl_pred * w_pl
    inp_fin = (pa_inp + lat_inp + pl_inp)/3
    return compute_acc(inp_fin, decode_lengths, targ, k)

# BLEU Metrics
def get_hyp_ref(input, targets, view, w_pa=0.333, w_lat=0.333, w_pl=0.333):
    pa_pred, lat_pred, pl_pred, decode_lengths, _ = input
    if view=="pa":
        _,pred_words = pa_pred.max(dim=-1)
    elif view=="lat":
        _,pred_words = lat_pred.max(dim=-1)
    elif view=="pl":
        _,pred_words = pl_pred.max(dim=-1)
    else:
        pa_inp = pa_pred * w_pa
        lat_inp = lat_pred * w_lat
        pl_inp = pl_pred * w_pl
        inp_fin = (pa_inp + lat_inp + pl_inp)/3
        _,pred_words = inp_fin.max(dim=-1)
    pred_words, ref = cp.deepcopy(list(pred_words)), targets.tolist()
    for x in range(len(ref)):
        numwords = len(ref[x])
        y=0
        while y < numwords:
            if ref[x][y] in {vocab.index('<start>'),
                            vocab.index('<end>'),
                            vocab.index('xxunk'), 
                            vocab.index('xxpad'), 
                            vocab.index('xxbos'), 
                            vocab.index('xxeos'), 
                            vocab.index('xxfld'), 
                            vocab.index('xxmaj'), 
                            vocab.index('xxup'), 
                            vocab.index('xxrep'), 
                            vocab.index('xxwrep'),
                            vocab.index('xxfake')}:
                del ref[x][y]
                numwords = len(ref[x])
            else:
                ref[x][y]=vocab[ref[x][y]]
                y+=1
    hypotheses = list()
    for i,cap in enumerate(pred_words): 
        hypotheses.append([x for x in cap.tolist()[:decode_lengths[i]] 
                           if x not in {vocab.index('<start>'),
                                        vocab.index('<end>'),
                                        vocab.index('xxunk'), 
                                        vocab.index('xxpad'), 
                                        vocab.index('xxbos'), 
                                        vocab.index('xxeos'), 
                                        vocab.index('xxfld'), 
                                        vocab.index('xxmaj'), 
                                        vocab.index('xxup'), 
                                        vocab.index('xxrep'), 
                                        vocab.index('xxwrep'),
                                        vocab.index('xxfake')}])
    for x in range(len(hypotheses)):
        for y in range(len(hypotheses[x])):
            hypotheses[x][y]=vocab[hypotheses[x][y]]
    assert len(ref) == len(hypotheses)
    return ref, hypotheses

def bleu1_pa(input, targets, w_pa=0.333, w_lat=0.333, w_pl=0.333): 
    ref, hypotheses = get_hyp_ref(input, targets, "pa")
    bleu1 = corpus_bleu(ref, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu1

def bleu1_lat(input, targets, w_pa=0.333, w_lat=0.333, w_pl=0.333): 
    ref, hypotheses = get_hyp_ref(input, targets, "lat")
    bleu1 = corpus_bleu(ref, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu1

def bleu1_pl(input, targets, w_pa=0.333, w_lat=0.333, w_pl=0.333): 
    ref, hypotheses = get_hyp_ref(input, targets, "pl")
    bleu1 = corpus_bleu(ref, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu1

def bleu1_weighted(input, targets, w_pa=0.333, w_lat=0.333, w_pl=0.333): 
    ref, hypotheses = get_hyp_ref(input, targets, "weighted", w_pa, w_lat, w_pl)
    bleu1 = corpus_bleu(ref, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu1

def bleu2_pa(input, targets, w_pa=0.333, w_lat=0.333, w_pl=0.333): 
    ref, hypotheses = get_hyp_ref(input, targets, "pa")
    bleu2 = corpus_bleu(ref, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    return bleu2

def bleu2_lat(input, targets, w_pa=0.333, w_lat=0.333, w_pl=0.333): 
    ref, hypotheses = get_hyp_ref(input, targets, "lat")
    bleu2 = corpus_bleu(ref, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    return bleu2

def bleu2_pl(input, targets, w_pa=0.333, w_lat=0.333, w_pl=0.333): 
    ref, hypotheses = get_hyp_ref(input, targets, "pl")
    bleu2 = corpus_bleu(ref, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    return bleu2

def bleu2_weighted(input, targets, w_pa=0.333, w_lat=0.333, w_pl=0.333):
    ref, hypotheses = get_hyp_ref(input, targets, "weighted", w_pa, w_lat, w_pl)
    bleu2 = corpus_bleu(ref, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    return bleu2

def bleu3_pa(input, targets, w_pa=0.333, w_lat=0.333, w_pl=0.333): 
    ref, hypotheses = get_hyp_ref(input, targets, "pa")
    bleu3 = corpus_bleu(ref, hypotheses, weights=(0.5, 0.5, 0, 0))
    return bleu3

def bleu3_lat(input, targets, w_pa=0.333, w_lat=0.333, w_pl=0.333): 
    ref, hypotheses = get_hyp_ref(input, targets, "lat")
    bleu3 = corpus_bleu(ref, hypotheses, weights=(0.5, 0.5, 0, 0))
    return bleu3

def bleu3_pl(input, targets, w_pa=0.333, w_lat=0.333, w_pl=0.333): 
    ref, hypotheses = get_hyp_ref(input, targets, "pl")
    bleu3 = corpus_bleu(ref, hypotheses, weights=(0.5, 0.5, 0, 0))
    return bleu3

def bleu3_weighted(input, targets, w_pa=0.333, w_lat=0.333, w_pl=0.333):
    ref, hypotheses = get_hyp_ref(input, targets, "weighted", w_pa, w_lat, w_pl)
    bleu3 = corpus_bleu(ref, hypotheses, weights=(0.5, 0.5, 0, 0))
    return bleu3

def bleu4_pa(input, targets, w_pa=0.333, w_lat=0.333, w_pl=0.333): 
    ref, hypotheses = get_hyp_ref(input, targets, "pa")
    bleu4 = corpus_bleu(ref, hypotheses, weights=(1, 0, 0, 0))
    return bleu4

def bleu4_lat(input, targets, w_pa=0.333, w_lat=0.333, w_pl=0.333): 
    ref, hypotheses = get_hyp_ref(input, targets, "lat")
    bleu4 = corpus_bleu(ref, hypotheses, weights=(1, 0, 0, 0))
    return bleu4

def bleu4_pl(input, targets, w_pa=0.333, w_lat=0.333, w_pl=0.333): 
    ref, hypotheses = get_hyp_ref(input, targets, "pl")
    bleu4 = corpus_bleu(ref, hypotheses, weights=(1, 0, 0, 0))
    return bleu4

def bleu4_weighted(input, targets, w_pa=0.333, w_lat=0.333, w_pl=0.333):
    ref, hypotheses = get_hyp_ref(input, targets, "weighted", w_pa, w_lat, w_pl)
    bleu4 = corpus_bleu(ref, hypotheses, weights=(1, 0, 0, 0))
    return bleu4

# Rouge_l Metrics
def compute_rouge_l(ref, hyp):
    rouge=0
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    for sentences in range(len(hyp)):
        rouge += scorer.score(" ".join(hyp[sentences]), " ".join(ref[sentences]))['rougeL'][2]
    return rouge/len(hyp)

def rouge_l_pa(input, targets, w_pa=0.333, w_lat=0.333, w_pl=0.333): 
    ref, hypotheses = get_hyp_ref(input, targets, "pa")
    return compute_rouge_l(ref, hypotheses)

def rouge_l_lat(input, targets, w_pa=0.333, w_lat=0.333, w_pl=0.333): 
    ref, hypotheses = get_hyp_ref(input, targets, "lat")
    return compute_rouge_l(ref, hypotheses)

def rouge_l_pl(input, targets, w_pa=0.333, w_lat=0.333, w_pl=0.333): 
    ref, hypotheses = get_hyp_ref(input, targets, "pl")
    return compute_rouge_l(ref, hypotheses)

def rouge_l_weighted(input, targets, w_pa=0.333, w_lat=0.333, w_pl=0.333): 
    ref, hypotheses = get_hyp_ref(input, targets, "weighted", w_pa, w_lat, w_pl)
    return compute_rouge_l(ref, hypotheses)

def split_model_all(arch):
    return L(arch.pa_model.encoder, arch.lat_model.encoder, arch.pa_model.decoder, arch.lat_model.decoder).map(params)
    
def beam_search_all(mod, img, img1, vocab = None, beam_size = 5):
    with torch.no_grad():
        k = beam_size
        
        ## imput tensor preparation
        img = img.unsqueeze(0) #treating as batch of size 1
        img1 = img1.unsqueeze(0)

        # encoder output
        encoder_out = mod.pa_model.encoder(img)
        encoder_out1 = mod.lat_model.encoder(img1)
        encoder_dim = encoder_out.size(-1)
        num_pixels = encoder_out.size(1)

        # expand or repeat 'k' time
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
        encoder_out1 = encoder_out1.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[vocab.index('<start>')]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Tensor to store top k sequences' alphas; now they're just 1s
        seqs_alpha = torch.ones(k, 1, 14, 14).to(device)  # (k, 1, enc_image_size, enc_image_size)
        
        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_alpha = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = mod.pa_model.decoder.init_hidden_state(encoder_out)
        h1, c1 = mod.lat_model.decoder.init_hidden_state(encoder_out1)

        hypotheses = list()

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            #Pa
            embeddings = mod.pa_model.decoder.embedding(k_prev_words).squeeze(1).float()  # (s, embed_dim)
            awe, alpha = mod.pa_model.decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
            alpha = alpha.view(-1, 14, 14)  # (s, enc_image_size, enc_image_size)
            gate = mod.pa_model.decoder.sigmoid(mod.pa_model.decoder.f_beta(h))
            awe = (gate * awe)

            h, c = mod.pa_model.decoder.lstm(torch.cat([embeddings, awe], dim=1), (h, c))
            scores = mod.pa_model.decoder.fc(h)
            scores = F.log_softmax(scores, dim=1)
            
            #Lat
            embeddings1 = mod.lat_model.decoder.embedding(k_prev_words).squeeze(1).float()  # (s, embed_dim)
            awe1, alpha1 = mod.lat_model.decoder.attention(encoder_out1, h1)  # (s, encoder_dim), (s, num_pixels)
            alpha1 = alpha1.view(-1, 14, 14)  # (s, enc_image_size, enc_image_size)
            gate1 = mod.lat_model.decoder.sigmoid(mod.lat_model.decoder.f_beta(h1))
            awe1 = (gate1 * awe1)

            h1, c1 = mod.lat_model.decoder.lstm(torch.cat([embeddings1, awe1], dim=1), (h1, c1))
            scores1 = mod.lat_model.decoder.fc(h1)
            scores1 = F.log_softmax(scores1, dim=1)
            
            #Combine scores
            pa_logits = glb_pa_logits[0]   # Only grabbling weights, not bias'
            lat_logits = glb_lat_logits[0]   # Only grabbling weights, not bias'
            mixed = torch.cat((pa_logits, lat_logits), dim=1)
            mixed_pred = mod.mixed_cls(mixed)
        
            # Add scores to prev scores
            mixed_pred = top_k_scores.expand_as(mixed_pred) + mixed_pred  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = mixed_pred[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = mixed_pred.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // len(vocab)  # (s)
            next_word_inds = top_k_words % len(vocab)  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1) stroes indices of words
            alpha = alpha.add(alpha1) # addition of two tensors
            alpha = alpha/2 # mean of output tensors
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)
            
            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                next_word != vocab.index('<end>')]

            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            h1 = h1[prev_word_inds[incomplete_inds]]
            c1 = c1[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            encoder_out1 = encoder_out1[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1
    
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    # Hypotheses
    hypotheses.append([w for w in seq if w not in {vocab.index('<start>'),
                                                    vocab.index('<end>'),
                                                    vocab.index('xxunk'), 
                                                    vocab.index('xxpad'), 
                                                    vocab.index('xxbos'), 
                                                    vocab.index('xxeos'), 
                                                    vocab.index('xxfld'), 
                                                    vocab.index('xxmaj'), 
                                                    vocab.index('xxup'), 
                                                    vocab.index('xxrep'), 
                                                    vocab.index('xxwrep'),
                                                    vocab.index('xxfake')}])
    return hypotheses, alphas

def visualize_att_all(image_path, image_path1, seq, alphas, path, smooth=True):
    """
    Visualizes caption with weights at every word.
    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    alphas = torch.FloatTensor(alphas)
    image = Image.open(path/image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)
    image1 = Image.open(path/image_path1)
    image1 = image1.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = seq

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        plt.imshow(image1)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.savefig(path/'imgcap.png', dpi=300, bbox_inches='tight')

#Info Sum
classes=["Atelectasis", 
        "Cardiomegaly", 
        "Consolidation", 
        "Edema",
        "Enlarged_Cardiomediastinum", 
        "Fracture", 
        "Lung_Lesion", 
        "Lung_Opacity", 
        "No_Finding", 
        "Pleural_Effusion",
        "Pleural_Other",
        "Pneumonia",
        "Pneumothorax",
        "Support_Devices",
        "Other"]
original_size=460

#Vision
def v_dls_trainval(bs, size, path, view, workers=8):
    dblock = DataBlock(
        blocks=(ImageBlock(cls=PILImageBW), MultiCategoryBlock(encoded=True, vocab=classes)),
        get_x=ColReader(view),
        get_y=ColReader(classes),
        splitter=ColSplitter(-1), 
        item_tfms=Resize(original_size), 
        batch_tfms=aug_transforms(size=size))
    return dblock.dataloaders(path, bs=bs, workers=workers)

def v_dls_test(bs, size, path, view, workers=8):
    dblock = DataBlock(
        blocks=(ImageBlock(cls=PILImageBW), MultiCategoryBlock(encoded=True, vocab=classes)),
        get_x=ColReader(view),
        get_y=ColReader(classes),
        splitter=None, 
        item_tfms=Resize(original_size), 
        batch_tfms=aug_transforms(size=size))
    return dblock.dataloaders(path, bs=bs, workers=workers)

def v_dls_new(bs, size, path, view, workers=8):
    dblock = DataBlock(
        blocks=(ImageBlock(cls=PILImageBW)),
        get_x=ColReader(view),
        splitter=None, 
        item_tfms=Resize(original_size), 
        batch_tfms=aug_transforms(size=size))
    return dblock.dataloaders(path, bs=bs, workers=workers)

#Lang
def l_dls_trainval(bs, path, workers=8):
    dblock = DataBlock(blocks=TextBlock.from_df(text_cols='path', is_lm=True),
                              get_x=ColReader(cols='text'),
                              splitter=ColSplitter('split'))
    return dblock.dataloaders(path, bs=bs, workers=workers)

def l_dls_test(bs, path, workers=8):
    dblock = DataBlock(blocks=TextBlock.from_df(text_cols='path', is_lm=True),
                              get_x=ColReader(cols='text'),
                              splitter=None)
    return dblock.dataloaders(path, bs=bs, workers=workers)

#Text
def tc_dls_trainval(bs, path, seq_len, vocab, workers=8):
    dblock = DataBlock(blocks=(TextBlock.from_df(text_cols='path', vocab=vocab, seq_len=seq_len),
                                MultiCategoryBlock(encoded=True, vocab=classes)),
                              get_x=ColReader(cols='text'),
                              get_y=ColReader(cols=classes),
                              splitter=ColSplitter('split'))
    return dblock.dataloaders(path, bs=bs, workers=workers)

def tc_dls_test(bs, path, seq_len, vocab, workers=8):
    dblock = DataBlock(blocks=(TextBlock.from_df(text_cols='path', vocab=vocab, seq_len=seq_len),
                                MultiCategoryBlock(encoded=True, vocab=classes)),
                              get_x=ColReader(cols='text'),
                              get_y=ColReader(cols=classes),
                              splitter=None)
    return dblock.dataloaders(path, bs=bs, workers=workers)

#Tab
def t_dls_trainval(bs, path, workers=8):
    procs = [Categorify, FillMissing, Normalize]
    cond = (path.split==False)
    train_idx = np.where( cond)[0]
    valid_idx = np.where(~cond)[0]
    splits = (list(train_idx),list(valid_idx))
    cont, cat = cont_cat_split(path, 1, dep_var=classes)
    tp = TabularPandas(path, 
                       procs, 
                       [],
                       cont[:5], 
                       y_names=classes, 
                       y_block=MultiCategoryBlock(encoded=True, vocab=classes),
                       splits=splits)
    return tp.dataloaders(bs, workers=workers)

def t_dls_test(bs, path, workers=8):
    procs = [Categorify, FillMissing, Normalize]
    cont, cat = cont_cat_split(path, 1, dep_var=classes)
    tp = TabularPandas(path, 
                       procs, 
                       [],
                       cont[:5], 
                       y_names=classes, 
                       y_block=MultiCategoryBlock(encoded=True, vocab=classes),
                       splits=None)
    return tp.dataloaders(bs, workers=workers)

class MixedDLSum():
    def __init__(self, pa_dl:TfmdDL, lat_dl:TfmdDL, text_dl:TfmdDL, tab_dl:TabDataLoader, device=device):
        "Stores away `tab_dl` and `vis_dl`, and overrides `shuffle_fn`"
        self.device = device
        pa_dl.shuffle_fn = self.shuffle_fn
        lat_dl.shuffle_fn = self.shuffle_fn
        text_dl.shuffle_fn = self.shuffle_fn
        tab_dl.shuffle_fn = self.shuffle_fn
        self.dls = [pa_dl, lat_dl, text_dl, tab_dl]
        self.count = 0
        self.fake_l = _FakeLoader(self, False, 0, 0)
    
    def __len__(self): return len(self.dls[0])
        
    def shuffle_fn(self, idxs):
        "Generates a new `rng` based upon which `DataLoader` is called"
        if self.count == 0: # if we haven't generated an rng yet
            self.rng = self.dls[0].rng.sample(idxs, len(idxs))
            self.count += 1
            return self.rng
        else:
            self.count = 0
            return self.rng
        
    def to(self, device): self.device = device
    
    def __iter__(self):
        "Iterate over your `DataLoader`"
        for i in self.dls:
            i.fake_l.num_workers=0
        z = zip(*[_loaders[i.fake_l.num_workers==0](i.fake_l) for i in self.dls])
        for b in z:
            if self.device is not None: 
                b = to_device(b, self.device)
            batch = []
            batch.append(self.dls[0].after_batch(b[0][0]))
            batch.append(self.dls[1].after_batch(b[1][0]))
            batch.append(self.dls[2].after_batch(b[2][0]))
            batch.extend(self.dls[3].after_batch(b[3]))
            try: # In case the data is unlabelled
                batch.append(b[1][1])
                yield tuple(batch)
            except:
                yield tuple(batch)
    
    def one_batch(self):
        "Grab a batch from the `DataLoader`"
        with self.fake_l.no_multiproc(): res = first(self)
        if hasattr(self, 'it'): delattr(self, 'it')
        return res
    
    def show_batch(self):
        "Show a batch from multiple `DataLoaders`"
        for dl in self.dls:
            dl.show_batch(max_n=1)

def get_pa_logits_sum(self, inp, out):
    global glb_pa_logits_sum
    glb_pa_logits_sum = inp
    #return None

def get_lat_logits_sum(self, inp, out):
    global glb_lat_logits_sum
    glb_lat_logits_sum = inp
    #return None

def get_text_logits(self, inp, out):
    global glb_text_logits
    glb_text_logits = inp
    #return None
    
def get_tab_logits(self, inp, out):
    global glb_tab_logits
    glb_tab_logits = inp
    #return None

class All(nn.Module):
    def __init__(self, pa_model, lat_model, text_model, tab_model, num_classes=len(classes)): 
        super(All, self).__init__()
        self.pa_model = pa_model
        self.lat_model = lat_model
        self.text_model = text_model
        self.tab_model = tab_model
        
        self.mixed_cls = nn.Linear(1324, num_classes)
        self.pa_cls = nn.Linear(512, num_classes)
        self.lat_cls = nn.Linear(512, num_classes)
        self.text_cls = nn.Linear(150, num_classes)
        self.tab_cls = nn.Linear(150, num_classes)
        
        #self.print_handle = self.tab_model.layers[2][0].register_forward_hook(printnorm)
        self.pa_handle = list(list(self.pa_model.children())[1].children())[-1].register_forward_hook(get_pa_logits_sum)
        self.lat_handle = list(list(self.lat_model.children())[1].children())[-1].register_forward_hook(get_lat_logits_sum)
        self.text_handle = self.text_model[-1].layers[-1].register_forward_hook(get_text_logits)
        self.tab_handle = self.tab_model.layers[2][0].register_forward_hook(get_tab_logits)
    
    def remove_my_hooks(self):
        self.pa_handle.remove()
        self.lat_handle.remove()
        self.text_handle.remove()
        self.tab_handle.remove()
        #self.print_handle.remove()
        return None
        
    def forward(self, x_pa, x_lat, x_text, x_cat, x_cont, label): 
        # PA Classifier
        if list(x_pa.shape)[1]!=3:
            x_pa = torch.cat((torch.as_tensor(x_pa),)*3, axis=1)
        pa_pred = self.pa_model(x_pa)
        
        # Lat Classifier
        if list(x_lat.shape)[1]!=3:
            x_lat = torch.cat((torch.as_tensor(x_lat),)*3, axis=1)
        lat_pred = self.lat_model(x_lat)

        # Text Classifier
        text_pred = self.text_model(x_text)        
        
        # Tabular Classifier
        tab_pred = self.tab_model(x_cat, x_cont)
        
        # Logits
        pa_logits = glb_pa_logits_sum[0]   # Only grabbling weights, not bias'
        lat_logits = glb_lat_logits_sum[0]   # Only grabbling weights, not bias'
        text_logits = glb_text_logits[0]   # Only grabbling weights, not bias'
        tab_logits = glb_tab_logits[0]   # Only grabbling weights, not bias'
        mixed = torch.cat((pa_logits, lat_logits, text_logits, tab_logits), dim=1)

        # Mixed Classifier
        mixed_pred = self.mixed_cls(mixed)
        return (pa_pred, lat_pred, text_pred, tab_pred, mixed_pred)

def ModCELossSum(pred, targ, ce=False):
    if type(pred)==tuple:
        pred = pred[0]
    if pred.shape!=targ.shape:
        pred = pred[0]
    pred = pred.sigmoid()
    if ce:
        loss = F.cross_entropy(pred, targ)
    else:
        loss = F.binary_cross_entropy_with_logits(pred, targ)
    return loss

class myGradientBlendingSum(nn.Module):
    def __init__(self, pa_weight=0.0, lat_weight=0.0, text_weight=0.0, tab_weight=0.0, all_weight=1.0, loss_scale=1.0, use_cel=False):
        "Expects weights for each model, the combined model, and an overall scale"
        super(myGradientBlendingSum, self).__init__()
        self.pa_weight = pa_weight
        self.lat_weight = lat_weight
        self.text_weight = text_weight
        self.tab_weight = tab_weight
        self.all_weight = all_weight
        self.ce = use_cel
        self.scale = loss_scale
        
    def forward(self, xb, yb):
        pa_out, lat_out, text_out, tab_out, all_out = xb
        targ = yb
        "Gathers `self.loss` for each model, weighs, then sums"
        pa_loss = ModCELossSum(pa_out, targ, self.ce) * self.scale
        lat_loss = ModCELossSum(lat_out, targ, self.ce) * self.scale
        text_loss = ModCELossSum(text_out, targ, self.ce) * self.scale
        tab_loss = ModCELossSum(tab_out, targ, self.ce) * self.scale
        all_loss = ModCELossSum(all_out, targ, self.ce) * self.scale
        
        weighted_pa_loss = pa_loss * self.pa_weight
        weighted_lat_loss = lat_loss * self.lat_weight
        weighted_text_loss = text_loss * self.text_weight
        weighted_tab_loss = tab_loss * self.tab_weight
        weighted_all_loss = all_loss * self.all_weight
        
        loss = weighted_pa_loss + weighted_lat_loss + weighted_text_loss + weighted_tab_loss + weighted_all_loss
        return loss

"""
FBetaMulti(beta=2, thresh=0.5, average='weighted'), #Only without CutMix
JaccardMulti(thresh=0.5, average='weighted'),
PrecisionMulti(thresh=0.5, average='weighted'),
RecallMulti(thresh=0.5, average='weighted'),
"""
#Accuracy
def pa_accuracy(inp, targ, thresh=0.5):
    pred,targ = flatten_check(inp[0].sigmoid(), targ)
    return accuracy_multi(pred, targ, thresh=thresh)

def lat_accuracy(inp, targ, thresh=0.5):
    pred,targ = flatten_check(inp[1].sigmoid(), targ)
    return accuracy_multi(pred, targ, thresh=thresh)

def text_accuracy(inp, targ, thresh=0.5):
    pred,targ = flatten_check(inp[2][0].sigmoid(), targ)
    return accuracy_multi(pred, targ, thresh=thresh)

def tab_accuracy(inp, targ, thresh=0.5):
    pred,targ = flatten_check(inp[3].sigmoid(), targ)
    return accuracy_multi(pred, targ, thresh=thresh)

def all_accuracy(inp, targ, thresh=0.5):
    pred,targ = flatten_check(inp[4].sigmoid(), targ)
    return accuracy_multi(pred, targ, thresh=thresh)

def weighted_accuracy_sum(inp, targ, thresh=0.5, pa_w=0.2, lat_w=0.2, text_w=0.2, tab_w=0.2, all_w=0.2):
    pa_inp = inp[0] * pa_w
    lat_inp = inp[1] * lat_w
    text_inp = inp[2][0] * text_w
    tab_inp = inp[3] * tab_w
    all_inp = inp[4] * all_w
    inp_fin = (pa_inp + lat_inp + text_inp + tab_inp + all_inp)/5
    pred,targ = flatten_check(inp_fin.sigmoid(), targ)
    return accuracy_multi(pred, targ, thresh=thresh)

#AP
def pa_ap(inp, targ):
    pred,targ = flatten_check(inp[0], targ)
    return APScoreMulti(average='weighted')(pred, targ)

def lat_ap(inp, targ):
    pred,targ = flatten_check(inp[1], targ)
    return APScoreMulti(average='weighted')(pred, targ)

def text_ap(inp, targ):
    pred,targ = flatten_check(inp[2][0], targ)
    return APScoreMulti(average='weighted')(pred, targ)

def tab_ap(inp, targ):
    pred,targ = flatten_check(inp[3], targ)
    return APScoreMulti(average='weighted')(pred, targ)

def all_ap(inp, targ):
    pred,targ = flatten_check(inp[4], targ)
    return APScoreMulti(average='weighted')(pred, targ)

def weighted_ap(inp, targ, pa_w=0.2, lat_w=0.2, text_w=0.2, tab_w=0.2, all_w=0.2):
    pa_inp = inp[0] * pa_w
    lat_inp = inp[1] * lat_w
    text_inp = inp[2][0] * text_w
    tab_inp = inp[3] * tab_w
    all_inp = inp[4] * all_w
    inp_fin = (pa_inp + lat_inp + text_inp + tab_inp + all_inp)/5
    pred,targ = flatten_check(inp_fin.sigmoid(), targ)
    return APScoreMulti(average='weighted')(pred, targ)

#ROC
def pa_roc(inp, targ):
    pred,targ = flatten_check(inp[0], targ)
    return RocAucMulti(average='weighted')(pred, targ)

def lat_roc(inp, targ):
    pred,targ = flatten_check(inp[1], targ)
    return RocAucMulti(average='weighted')(pred, targ)

def text_roc(inp, targ):
    pred,targ = flatten_check(inp[2][0], targ)
    return RocAucMulti(average='weighted')(pred, targ)

def tab_roc(inp, targ):
    pred,targ = flatten_check(inp[3], targ)
    return RocAucMulti(average='weighted')(pred, targ)

def all_roc(inp, targ):
    pred,targ = flatten_check(inp[4], targ)
    return RocAucMulti(average='weighted')(pred, targ)

def weighted_roc(inp, targ, pa_w=0.2, lat_w=0.2, text_w=0.2, tab_w=0.2, all_w=0.2):
    pa_inp = inp[0] * pa_w
    lat_inp = inp[1] * lat_w
    text_inp = inp[2][0] * text_w
    tab_inp = inp[3] * tab_w
    all_inp = inp[4] * all_w
    inp_fin = (pa_inp + lat_inp + text_inp + tab_inp + all_inp)/5
    pred,targ = flatten_check(inp_fin.sigmoid(), targ)
    return RocAucMulti(average='weighted')(pred, targ)

class CutMixAll(Callback):
    "Implementation of `https://arxiv.org/abs/1905.04899`"
    run_after,run_valid = [Normalize],False
    def __init__(self, alpha=1.): self.distrib = Beta(tensor(alpha), tensor(alpha))
        
    def before_fit(self):
        self.stack_y = getattr(self.learn.loss_func, 'y_int', False)
        if self.stack_y: self.old_lf,self.learn.loss_func = self.learn.loss_func,self.lf

    def after_fit(self):
        if self.stack_y: self.learn.loss_func = self.old_lf
            
    def before_batch(self):
        W, H = self.xb[0].size(3), self.xb[0].size(2)
        lam = self.distrib.sample((1,)).squeeze().to(self.x[0].device)
        lam = torch.stack([lam, 1-lam])
        self.lam = lam.max()
        shuffle = torch.randperm(self.y.size(0)).to(self.x[0].device)
        xb1,self.yb1 = tuple(L(self.xb[0]).itemgot(shuffle)),tuple(L(self.yb).itemgot(shuffle))
        nx_dims = len(self.x[0].size())
        
        W1, H1 = self.xb[1].size(3), self.xb[1].size(2)
        lam1 = self.distrib.sample((1,)).squeeze().to(self.x[1].device)
        lam1 = torch.stack([lam1, 1-lam1])
        self.lam1 = lam1.max()
        shuffle1 = torch.randperm(self.y.size(0)).to(self.x[1].device)
        xb1a,self.yb1a = tuple(L(self.xb[1]).itemgot(shuffle1)),tuple(L(self.yb).itemgot(shuffle1))
        nx_dims1 = len(self.x[1].size())
        
        x1, y1, x2, y2, x1a, y1a, x2a, y2a = self.rand_bbox(W, H, self.lam, W1, H1, self.lam1)
        self.learn.xb[0][:, :, x1:x2, y1:y2] = xb1[0][:, :, x1:x2, y1:y2]
        self.lam = (1 - ((x2-x1)*(y2-y1))/float(W*H)).item()
        
        self.learn.xb[1][:, :, x1a:x2a, y1a:y2a] = xb1a[0][:, :, x1a:x2a, y1a:y2a]
        self.lam1 = (1 - ((x2a-x1a)*(y2a-y1a))/float(W1*H1)).item()
    
    def lf(self, pred, *yb):
        if not self.training: return self.old_lf(pred, *yb)
        with NoneReduce(self.old_lf) as lf:
            loss = torch.lerp(lf(pred,*self.yb1), lf(pred,*yb), self.lam)
            loss1 = torch.lerp(lf(pred,*self.yb1a), lf(pred,*yb), self.lam1)
        return reduce_loss((loss+loss1)/2, getattr(self.old_lf, 'reduction', 'mean'))

    def rand_bbox(self, W, H, lam, W1, H1, lam1):
        cut_rat = torch.sqrt(1. - lam)
        cut_w = (W * cut_rat).type(torch.long)
        cut_h = (H * cut_rat).type(torch.long)
        # uniform
        cx = torch.randint(0, W, (1,)).to(self.x[0].device)
        cy = torch.randint(0, H, (1,)).to(self.x[0].device)
        x1 = torch.clamp(cx - cut_w // 2, 0, W)
        y1 = torch.clamp(cy - cut_h // 2, 0, H)
        x2 = torch.clamp(cx + cut_w // 2, 0, W)
        y2 = torch.clamp(cy + cut_h // 2, 0, H)
        
        cut_rat1 = torch.sqrt(1. - lam1)
        cut_w1 = (W1 * cut_rat1).type(torch.long)
        cut_h1 = (H1 * cut_rat1).type(torch.long)
        # uniform
        cx1 = torch.randint(0, W1, (1,)).to(self.x[1].device)
        cy1 = torch.randint(0, H1, (1,)).to(self.x[1].device)
        x1a = torch.clamp(cx1 - cut_w1 // 2, 0, W1)
        y1a = torch.clamp(cy1 - cut_h1 // 2, 0, H1)
        x2a = torch.clamp(cx1 + cut_w1 // 2, 0, W1)
        y2a = torch.clamp(cy1 + cut_h1 // 2, 0, H1)
        return x1, y1, x2, y2, x1a, y1a, x2a, y2a

def split_model_sum(arch):
    return L(arch.pa_model[0][:6], 
             arch.lat_model[0][:6], 
             arch.pa_model[0][6:], 
             arch.lat_model[0][6:], 
             arch.text_model[0], 
             arch.tab_model.embeds,
             arch.text_model[1],
             arch.pa_model[1:], 
             arch.lat_model[1:]).map(params)

workers=0
seq_len=72
def predict_sum(learn, pa, lat, text, tab, label, df, tabcols, size):
    new = pd.DataFrame(columns=df.columns)
    new.loc[0, 'path'] = text
    new.loc[0, 'images'] = pa
    new.loc[0, 'images1'] = lat

    for column in tabcols:
        new.loc[0, column]=tab.loc[column]

    for column in classes:
        new.loc[0, column]=label[classes.index(column)]

    new = pd.concat([new, new])
    new.reset_index(drop=True, inplace=True)
    new.iloc[1].path = str(new.iloc[1].path + new.iloc[1].path)

    tofloat = tabcols
    tofloat.extend(classes)
    new = pd.concat((new[['path', 'images', 'images1', 'ViewPosition']], new[tofloat].apply(lambda col: pd.to_numeric(col, errors='coerce'))), axis=1)
    bs=1

    pa_dls_new_sum = v_dls_test(bs, size, new, 'images', workers)
    lat_dls_new_sum = v_dls_test(bs, size, new, 'images1', workers)
    text_class_dls_new = tc_dls_test(bs, new, seq_len, vocab, workers)
    tab_dls_new = t_dls_test(bs, new, workers)

    mixed_dls_sum_new = MixedDLSum(pa_dls_new_sum[0], lat_dls_new_sum[0], text_class_dls_new[0], tab_dls_new[0])
    res, y = learn.get_preds(dl=mixed_dls_sum_new)
    return new, res[4].sigmoid()

def return_grad(learn, mod, img, pa, path, size, clsidxs):
    class Hook():
        def __init__(self, m):
            self.hook = m.register_forward_hook(self.hook_func)   
        def hook_func(self, m, i, o): self.stored = o.detach().clone()
        def __enter__(self, *args): return self
        def __exit__(self, *args): self.hook.remove()
    with Hook(mod[0]) as hook:
        with torch.no_grad(): output = mod.eval()(img)
        act = hook.stored
        a = torch.sigmoid(output)>0.5
        class_idxes = [i for i, val in enumerate(a[0]) if val] 
        if class_idxes==[]:
            class_idxes.append(classes.index('Other'))
        class_idxes=clsidxs
        class HookBwd():
            def __init__(self, m):
                self.hook = m.register_backward_hook(self.hook_func)   
            def hook_func(self, m, gi, go): self.stored = go[0].detach().clone()
            def __enter__(self, *args): return self
            def __exit__(self, *args): self.hook.remove()
        def cmap(class_idx):
            with HookBwd(mod[0]) as hookg:
                with Hook(mod[0]) as hook:
                    output = mod.eval()(img)
                    act = hook.stored
                output[0,class_idx].backward()
                grad = hookg.stored
            return act, grad
        i=1
        for idx in class_idxes:
            act, grad = cmap(idx)
            w = grad[0].mean(dim=[1,2], keepdim=True)
            cam_map = (w * act[0]).sum(0)
            x_dec = TensorImage(learn.dls.dls[pa].decode((img,))[0][0])
            f,ax = plt.subplots()
            x_dec.show(ctx=ax)
            ax.imshow(cam_map.detach().cpu(), alpha=0.6, extent=(0,size,size,0),
                        interpolation='bilinear', cmap='magma');
            if not pa:
                f.savefig(path/str("pa_gradcam"+str(i)+".png"))
            else:
                f.savefig(path/str("lat_gradcam"+str(i)+".png"))
            i+=1