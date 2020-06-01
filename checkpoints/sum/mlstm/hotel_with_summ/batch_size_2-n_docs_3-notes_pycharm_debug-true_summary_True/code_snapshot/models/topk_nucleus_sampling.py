import torch.nn.functional as F
import torch

SOS_token="<DOC>"

UNK_token='<OOV>'

EOS_token="</DOC>"

PAD_token='<pad>' # SOS= Start-of-sentence token


def to_var(x, requires_grad):
    return x.clone().detach().requires_grad_(requires_grad)

def subsequent_mask(size):
    #"Mask out subsequent positions.”
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class TopKNucleusSampling:
    def __init__(self, decoder, generator, idx2word:dict, word2idx:dict, temperature, top_k, top_p, repetition_penalty, device='cpu',  min_length=3, max_length=15, start_token_id=SOS_token, end_token_id=EOS_token, pad_token_id=PAD_token,):
        super(TopKNucleusSampling, self).__init__()
        self.decoder = decoder
        self.generator = generator

        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.pad_token_id = pad_token_id

        self.min_length = min_length
        self.max_length = max_length

        self.idx2word = idx2word
        self.word2idx = word2idx
        self.device = device

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def decode(self, decoder_hidden, outU, outB, outA, predExp):
        batch_size = decoder_hidden.size(1 if self.generator is None else 0)
        decoder_input = torch.stack([torch.LongTensor([SOS_token]) for _ in range(batch_size)], dim=1)
        decoder_input = decoder_input.to(self.device)

        decoder_length = torch.LongTensor([0 for _ in range(batch_size)])
        decoder_length = decoder_length.to(self.device)

        for n in range(self.max_length):
            decoder_length += 1

            if self.generator is None:
                decoder_output, _, _, _, _ = self.decoder(decoder_input, decoder_length, decoder_hidden, outU, outB, outA, predExp)
                logits = decoder_output.transpose(0, 1)[:, -1, :]
            else:
                decoder_input = decoder_input.transpose(0, 1)
                out = self.decoder.decode(decoder_hidden.unsqueeze(1), to_var(decoder_input, requires_grad=False).to(self.device), to_var(subsequent_mask(decoder_input.size(1)).long(), requires_grad=False).to(self.device))
                logits = self.generator(out[:, -1])
                decoder_input = decoder_input.transpose(0, 1)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            if self.repetition_penalty != 1.0:
                for i in range(decoder_input.shape[1]):
                    for j in set(decoder_input[:, i].tolist()):
                        logits[i, j] /= self.repetition_penalty

            logits = logits / self.temperature
            if n < self.min_length:
                logits[:, self.end_token_id] = -1e20

            filtered_logits = self.top_k_top_p_filtering(logits)
            probabilities = F.softmax(filtered_logits, dim=-1)

            if self.temperature == 0:  # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(probabilities, num_samples=1)

            decoder_input = torch.cat((decoder_input, next_token.transpose(0, 1)), dim=0)

        _sentences = [[] for _ in range(batch_size)]
        isFinished = [False for _ in range(batch_size)]
        for sentence in decoder_input.tolist()[1:]:
            for i, word_idx in enumerate(sentence):
                if not isFinished[i]:
                    if word_idx == EOS_token:
                        isFinished[i] = True
                    else:
                        _sentences[i].append(self.idx2word[int(word_idx)])

        return _sentences

    def top_k_top_p_filtering(self, logits):
        top_k = min(self.top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        elif self.top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # a quick question: at the beginning of the function it says "top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering)",
            # but later tokens with cumu prob >= top_p are removed instead?
            # cumulative_probs > top_p puts zero byte everywhere where sum will be less or equal than top_p. These zeroed categories will be actually removed later.
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p
            # Shift the indices to the right to keep also the first token above the threshold.
            # To avoid having no values
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

        logits[indices_to_remove] = -float('Inf')
        return logits

