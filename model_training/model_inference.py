

import torch
import torch.nn.functional as F
# from model_training.combined_train import GPT, GPTConfig
from train_model_small import GPT, GPTConfig
from tokenizers import Tokenizer






if __name__ == '__main__':

    max_length = 50


    tokenizer = Tokenizer.from_file('/Users/xai/Personal/Projects/TeluguLM/tokenization/telugu_tokenizer/tokenizer.json')
    text = "అతను అక్కడికి వెళ్ళాడు కానీ"
    # text = "ఎందరో మహానుభావులు అందరికి "
    tokens = tokenizer.encode(text)
    x = torch.tensor(tokens.ids)
    x = x.reshape((1, -1))

    # model_path = "model_checkpoint_87k.pt"
    model_path = "./models/model_checkpoint_47296.pt"

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)

    print(checkpoint['step'])
    print(checkpoint['loss'])

    config = GPTConfig(vocab_size=32768, block_size=512, n_layer=8, n_head=8, n_embd=512)
    model = GPT(config)
    model = torch.compile(model)
    model.load_state_dict(checkpoint['model_state_dict'])

    # torch.manual_seed(42)

    while x.size(1) < max_length:
        with torch.no_grad():
            logits, loss = model.forward(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # Do topk sampling with k=50
            topk_probs, topk_inds = torch.topk(probs, 50, -1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_inds, -1, ix)
            x = torch.cat((x, xcol), dim=1)

    for i in range(len(x)):
        token_ids = x[i].tolist()
        generated_tokens = tokenizer.decode(token_ids)
        print(generated_tokens)