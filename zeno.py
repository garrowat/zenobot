import json

# Flask stuff
from flask import Flask, url_for, make_response
from flask_classful import FlaskView, route

# FastAI, PyTorch, NLP stuff
from fastai.nlp import *
from fastai.lm_rnn import *
from fastai import sgdr
from torchtext import vocab, data

app = Flask(__name__)

# Make a representation for the flask view
def output_json(data, code, headers=None):
    content_type = 'application/json'
    dumped = json.dumps(data)
    if headers:
        headers.update({'Content-Type': content_type})
    else:
        headers = {'Content-Type': content_type}
    response = make_response(dumped, code, headers)
    return response

# Define LSTM model class
class CharSeqStatefulLSTM(nn.Module):
    def __init__(self, vocab_size, n_fac, bs, nl):
        super().__init__()
        self.vocab_size,self.nl = vocab_size,nl
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.LSTM(n_fac, n_hidden, nl, dropout=0.5)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        self.init_hidden(bs)
        
    def forward(self, cs, **kwargs):
        bs = cs[0].size(0)
        if self.h[0].size(1) != bs: self.init_hidden(bs)
        self.rnn.flatten_parameters()
        self.h = (self.h[0].cpu(), self.h[1].cpu())
        outp,h = self.rnn(self.e(cs), self.h)
        return F.log_softmax(self.l_out(outp), dim=-1).view(-1, self.vocab_size)
    
    def init_hidden(self, bs):
        self.h = (V(torch.zeros(self.nl, bs, n_hidden)),
                  V(torch.zeros(self.nl, bs, n_hidden)))

# Set up the paths
PATH='data/proverbs/'
TRN_PATH = 'train/'
VAL_PATH = 'valid/'
TRN = PATH + TRN_PATH
VAL = PATH + VAL_PATH

TEXT = data.Field(lower=True, tokenize=list)
bs=64; bptt=8; n_fac=42; n_hidden=128

FILES = dict(train=TRN_PATH, validation=VAL_PATH, test=VAL_PATH)
md = LanguageModelData.from_text_files(PATH, TEXT, **FILES, bs=bs, bptt=bptt, min_freq=3)

m = CharSeqStatefulLSTM(md.nt, n_fac, 256, 2)
m.load_state_dict(torch.load((PATH + 'models/gen_0_dict'), map_location=lambda storage, loc: storage))
m.eval()

# Predict the next character
def get_next(inp):
    idxs = TEXT.numericalize(inp, device=-1)
    pid = idxs.transpose(0,1)
    pid = pid.cpu()
    vpid = VV(pid)
    vpid = vpid.cpu()
    p = m(vpid)
    r = torch.multinomial(p[-1].exp(), 1)
    return TEXT.vocab.itos[to_np(r)[0]]

# get_next based on input string
def get_next_n(inp, n):
    res = inp
    for i in range(n):
        c = get_next(inp)
        res += c
        inp = inp[1:]+c
        if c == '.': break
    return res

# Define flask view for returning API requests
class ProverbsView(FlaskView):    
    route_base = '/zenobot'
    representations = {'application/json': output_json}

    @route('/proverb/<input>')
    def get_proverb(self, input):
        input = str(input)
        proverb = get_next_n(input + " ", 1000)
        return {'proverb': proverb}

ProverbsView.register(app)

if __name__ == "__main__":
    print(("Loading fastai..."))
    app.run(debug=True)