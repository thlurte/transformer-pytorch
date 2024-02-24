import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from dataset import BilingualDataset,casual_mask

from pathlib import Path

def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config,ds,lang):
    # tokenizer
    tokenizer_path= Path(config["tokenizer_file"].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer=Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer=Whitespace()
        trainer=WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer=Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer


def get_ds(config):
    ds_raw=load_dataset('opus_books',f'{config['lang_src']}-{config['lang_tgt']}',split='train')

    # Build Tokenizer
    tokenizer_src=get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_tgt=get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])

    # Train split
    train_ds_size=int(0.9*len(ds_raw))
    val_ds_size=len(ds_raw)-train_ds_size
    train_ds_raw,val_ds_raw=random_split(ds_raw,[train_ds_size,val_ds_size])

    train_ds=BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
    val_ds=BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])

    max_len_src=0
    max_len_tgt=0

    for item in ds_raw:
        src_ids=tokenizer_src.encode(item['translation'][config['lang_src']]).idx
        tgt_ids=tokenizer_src.encode(item['translation'][config['lang_tgt']]).idx
        max_len_src=max(src_ids,max_len_src)
        max_len_tgt=max(tgt_ids,max_len_tgt)

    print(f"The length of the source sentence {max_len_src}")
    print(f"The length of the source sentence {max_len_tgt}")

    train_dataloader=DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True)
    val_dataloader=DataLoader(val_ds,batch_size=1,shuffle=True)

    return train_dataloader,val_dataloader,tokenizer_src, tokenizer_tgt