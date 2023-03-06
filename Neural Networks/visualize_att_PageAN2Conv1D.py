
file_num = 72
file_page = 0

result = print_attention_AN2_inp3_sents(webpageEncoder, Xtest, x_test, file_num, file_page, tokenizer, MAX_SENTS, MAX_SENT_LENGTH)

results = [print_attention_AN_inp3_words(sentEncoder, Xtest, file_num, file_page, sent, tokenizer, MAX_SENT_LENGTH) for sent in range(len(result))]
