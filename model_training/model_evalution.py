import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from tokenizers import Tokenizer
from dotenv import load_dotenv
load_dotenv()



def calculate_perplexity(model, text, tokenizer, context_len=512):
    encoding = tokenizer.encode(text)
    ids = encoding.ids

    device = next(model.parameters()).device
    model.eval()

    log_probs = []
    with torch.no_grad():
        for i in range(1, len(ids)):
            context = ids[:i][-context_len:]
            inp_tokens = torch.tensor([context], device=device)

            logits = model(inp_tokens)[0]
            logits = logits[:, -1, :]

            log_prob = F.log_softmax(logits.float(), dim=-1)[0, ids[i]]
            log_probs.append(log_prob.item())

    log_probs = np.array(log_probs)
    perplexity = np.exp(-log_probs.mean())
    return perplexity



def perplexity_fast(model, text, tokenizer, context_len=512):

    ids = tokenizer.encode(text).ids
    device = next(model.parameters()).device

    x = torch.tensor([ids[-context_len:][:-1]], device=device)
    y = torch.tensor(ids[-context_len:][1:], device=device)

    with torch.no_grad():
        logits = model(x)[0]

    probabilities = F.log_softmax(logits.float(), dim=-1).squeeze()

    probs = probabilities[torch.arange(len(probabilities)), y]
    return torch.exp(-1*torch.mean(probs)).item()



if __name__ == '__main__':

    UPLOAD_MODEL = True

    # Load the tokenizer
    tokenizer = Tokenizer.from_file("../tokenization/telugu_tokenizer/telugu_unigram_tokenizer.json")

    text = """
తెలుగు భారతదేశంలో మాట్లాడబడే ప్రధాన ద్రావిడ భాషలలో ఒకటి. ఇది ముఖ్యంగా ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాలలో విస్తృతంగా ఉపయోగించబడుతుంది. తెలుగు భాషకు వేల సంవత్సరాల సాహిత్య సంప్రదాయం ఉంది. నన్నయ, తిక్కన, ఎర్రప్రగడ వంటి కవులు మహాభారతాన్ని తెలుగులో అనువదించి భాషకు అపారమైన కీర్తిని తెచ్చారు. కాలక్రమేణా తెలుగు సాహిత్యం కవిత్వం, నవలలు, నాటకాలు, వ్యాసాలు వంటి అనేక రూపాలలో అభివృద్ధి చెందింది.

ఇటీవల కాలంలో సాంకేతిక పరిజ్ఞానం వేగంగా అభివృద్ధి చెందుతోంది. కంప్యూటర్లు, స్మార్ట్‌ఫోన్లు మరియు ఇంటర్నెట్ వల్ల మన జీవిత విధానం పూర్తిగా మారిపోయింది. ప్రజలు ఇప్పుడు సమాచారాన్ని కొన్ని క్షణాల్లోనే పొందగలుగుతున్నారు. విద్య, ఆరోగ్యం, వ్యవసాయం వంటి రంగాలలో కూడా డిజిటల్ సాంకేతికత విస్తృతంగా ఉపయోగించబడుతోంది. కృత్రిమ మేధస్సు మరియు యంత్ర అధ్యయనం వంటి రంగాలు ప్రపంచవ్యాప్తంగా పరిశోధనలకు దారి తీస్తున్నాయి.

గ్రామీణ భారతదేశంలో వ్యవసాయం ప్రధాన జీవనాధారం. రైతులు పంటలపై ఆధారపడి జీవిస్తారు. వర్షపాతం సరైన సమయంలో పడితే పంటలు బాగా పండుతాయి. కానీ వాతావరణ మార్పుల కారణంగా కొన్ని ప్రాంతాల్లో అనిశ్చితి పెరిగింది. అందువల్ల నీటి నిర్వహణ, ఆధునిక వ్యవసాయ పద్ధతులు మరియు శాస్త్రీయ సలహాలు రైతులకు చాలా ఉపయోగపడుతున్నాయి.

పర్యావరణ పరిరక్షణ కూడా మన సమాజానికి చాలా ముఖ్యమైన అంశం. అడవుల నాశనం, కాలుష్యం మరియు వాతావరణ మార్పు వంటి సమస్యలు ప్రపంచవ్యాప్తంగా చర్చకు వస్తున్నాయి. చెట్లను నాటడం, నీటిని సంరక్షించడం మరియు పునర్వినియోగం వంటి చర్యలు పర్యావరణాన్ని రక్షించడానికి సహాయపడతాయి. ప్రతి వ్యక్తి తన బాధ్యతను గుర్తించి ప్రకృతిని కాపాడాలి.

విద్య సమాజ అభివృద్ధికి పునాది. మంచి విద్య ద్వారా వ్యక్తులు తమ జీవితాలను మెరుగుపరుచుకోగలరు. పాఠశాలలు మరియు విశ్వవిద్యాలయాలు విద్యార్థులకు జ్ఞానం మాత్రమే కాకుండా ఆలోచనా విధానాన్ని కూడా అందిస్తాయి. ఉపాధ్యాయులు విద్యార్థులను ప్రేరేపించి వారి ప్రతిభను వెలికితీసే ముఖ్యమైన పాత్ర పోషిస్తారు.

నేటి ప్రపంచంలో సమాచార ప్రసారం చాలా వేగంగా జరుగుతోంది. వార్తలు, వ్యాసాలు మరియు సామాజిక మాధ్యమాలు ప్రజల అభిప్రాయాలను ప్రభావితం చేస్తున్నాయి. సమాచారాన్ని జాగ్రత్తగా పరిశీలించడం మరియు నిజానిజాలను తెలుసుకోవడం ప్రతి పౌరుడి బాధ్యత. బాధ్యతాయుతమైన సమాచార వినియోగం సమాజానికి మేలు చేస్తుంది."""

    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_fol_path = "/Users/xai/Personal/Projects/TeluguLM/model_training/telugu_llama"
    # model_fol_path = "/Users/xai/Personal/Projects/TeluguLM/model_training/telugu-gpt/checkpoint-107300"

    tokenizer = AutoTokenizer.from_pretrained(model_fol_path)
    model = AutoModelForCausalLM.from_pretrained(model_fol_path)

    model.eval()

    encodings = tokenizer(text, return_tensors="pt")

    input_ids = encodings.input_ids

    # Calculate loss
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    # Perplexity
    perplexity = torch.exp(loss)

    print(f"Loss: {loss.item():.4f}")
    print(f"Perplexity: {perplexity.item():.4f}")

    if UPLOAD_MODEL:
        model.push_to_hub("harsha-desaraju/telugu-llama-small", commit_message="version-1", token=os.getenv("HF_TOKEN"))


