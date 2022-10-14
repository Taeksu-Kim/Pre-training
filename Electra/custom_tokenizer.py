import os
from tokenizers import BertWordPieceTokenizer
from transformers import ElectraTokenizerFast

def Build_Tokenizer(input_courpus_files,
                    tokenizer_save_dir,
                    vocab_size=35000,
                    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
                    model_max_input_len=512,
                    clean_text=True,
                    handle_chinese_chars=True,
                    strip_accents=False,
                    lowercase=False):

    if not os.path.exists(tokenizer_save_dir):
        print('tokenizer_save_dir {} is created.'.format(tokenizer_save_dir))
        os.makedirs(tokenizer_save_dir)

    # Initialize an empty tokenizer
    wp_tokenizer = BertWordPieceTokenizer(
        clean_text=clean_text,   # " ", "\t", "\n", "\r" 등의 공백 문자는 Token으로 하지 않고 제거. ["좋은"," ","예제"] -> ["좋은","예제"]
        handle_chinese_chars=handle_chinese_chars,  # 한자는 모두 char 단위로 분할
        strip_accents=strip_accents,    # True: [YehHamza] -> [Yep, Hamza]
        lowercase=lowercase,    # Hello -> hello
    )   

    wp_tokenizer.train(
        files=input_courpus_files,
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens,
        wordpieces_prefix="##"
    )

    wp_tokenizer.save_model(tokenizer_save_dir)

    print('vocab.txt is saved in {}'.format(tokenizer_save_dir))

    tokenizer = ElectraTokenizerFast(
      vocab_file=tokenizer_save_dir+'/vocab.txt',
      max_len=model_max_input_len,
      do_lower_case=lowercase)

    tokenizer.save_pretrained(tokenizer_save_dir)

    print('Electra Tokenizer is saved in {}'.format(tokenizer_save_dir))
