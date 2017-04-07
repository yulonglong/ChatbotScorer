import numpy as np
import logging

logger = logging.getLogger(__name__)

emb_reader = None

def create_model(args, overal_maxlen, vocab):

    import keras.backend as K
    from keras.layers.embeddings import Embedding
    from keras.models import Sequential, Model
    from keras.layers.core import Dense, Dropout, Activation
    from chatbotscorer.my_layers import Attention, MeanOverTime, Conv1DWithMasking, MaxPooling1DWithMasking

    #############################################################################
    ## Recurrence unit type
    #

    if args.recurrent_unit == 'lstm':
        from keras.layers.recurrent import LSTM as RNN
    elif args.recurrent_unit == 'gru':
        from keras.layers.recurrent import GRU as RNN
    elif args.recurrent_unit == 'simple':
        from keras.layers.recurrent import SimpleRNN as RNN

    ##############################################################################
    ## Create Model
    #
    
    default_dropout_W = args.dropout_rate
    default_dropout_U = args.dropout_rate / 5.0
    default_dropout = args.dropout_rate
    cnn_border_mode='same'

    if args.model_type == 'cnn':
        logger.info('Building a CNN model')
        assert (args.cnn_dim > 0)

        model = Sequential()
        from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
        sequence1 = Input(shape=(None,), dtype='int32')
        sequence2 = Input(shape=(None,), dtype='int32')
        embed1 = Embedding(args.vocab_size, args.emb_dim, mask_zero=True, name="Embedding1")(sequence1)
        embed2 = Embedding(args.vocab_size, args.emb_dim, mask_zero=True, name="Embedding2")(sequence2)
        
        conv1 = embed1
        conv2 = embed2

        for i in range(args.cnn_layer):
            conv1 = Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1)(conv1)
            conv2 = Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1)(conv2)

        merged = merge([conv1, conv2], mode='concat', concat_axis=1) # Concatenate the question and the answer by length (number of words)

        # if pooling type is not specified, use attsum by default
        if not args.pooling_type:
            args.pooling_type = 'attsum'

        if args.pooling_type == 'meanot':
            pooled = MeanOverTime(mask_zero=True)(merged)
        elif args.pooling_type.startswith('att'):
            pooled = Attention(op=args.pooling_type, name="attention_layer")(merged)
        logger.info('%s pooling layer added!', args.pooling_type)

        densed = Dense(1)(pooled)
        score = Activation('sigmoid')(densed)
        model = Model(input=[sequence1, sequence2], output=score)
        
        # get the WordEmbedding layer index
        model.emb_index = []
        model_layer_index = 0
        for test in model.layers:
            if (test.name in {'Embedding1', 'Embedding2'}):
                model.emb_index.append(model_layer_index)
            model_layer_index += 1

    elif args.model_type == 'rnn':
        logger.info('Building a RNN model')
        assert (args.rnn_dim > 0)

        model = Sequential()
        from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
        sequence1 = Input(shape=(None,), dtype='int32')
        sequence2 = Input(shape=(None,), dtype='int32')
        embed1 = Embedding(args.vocab_size, args.emb_dim, mask_zero=True, name="Embedding1")(sequence1)
        embed2 = Embedding(args.vocab_size, args.emb_dim, mask_zero=True, name="Embedding2")(sequence2)

        rnn1 = embed1
        rnn2 = embed2
        for i in range(args.rnn_layer):
            rnn1 = LSTM(args.rnn_dim, return_sequences=True, dropout_W=default_dropout_W, dropout_U=default_dropout_U)(rnn1)
            rnn2 = LSTM(args.rnn_dim, return_sequences=True, dropout_W=default_dropout_W, dropout_U=default_dropout_U)(rnn2)

        merged = merge([rnn1, rnn2], mode='concat', concat_axis=1) # Concatenate the question and the answer by length (number of words)

        # if pooling type is not specified, use attsum by default
        if not args.pooling_type:
            args.pooling_type = 'attsum'

        if args.pooling_type == 'meanot':
            pooled = MeanOverTime(mask_zero=True)(merged)
        elif args.pooling_type.startswith('att'):
            pooled = Attention(op=args.pooling_type, name="attention_layer")(merged)
        logger.info('%s pooling layer added!', args.pooling_type)

        densed = Dense(1)(pooled)
        score = Activation('sigmoid')(densed)
        model = Model(input=[sequence1, sequence2], output=score)
        
        # get the WordEmbedding layer index
        model.emb_index = []
        model_layer_index = 0
        for test in model.layers:
            if (test.name in {'Embedding1', 'Embedding2'}):
                model.emb_index.append(model_layer_index)
            model_layer_index += 1
            
    elif args.model_type == 'brnn' or args.model_type == 'brnn2':
        logger.info('Building a Bidirectional RNN model')
        assert (args.rnn_dim > 0)

        model = Sequential()
        from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
        sequence1 = Input(shape=(None,), dtype='int32')
        sequence2 = Input(shape=(None,), dtype='int32')
        embed1 = Embedding(args.vocab_size, args.emb_dim, mask_zero=True, name="Embedding1")(sequence1)
        embed2 = Embedding(args.vocab_size, args.emb_dim, mask_zero=True, name="Embedding2")(sequence2)

        rnn1forward = embed1
        rnn1backward = embed1
        rnn2forward = embed2
        rnn2backward = embed2
        for i in range(args.cnn_layer):
            rnn1forward = LSTM(args.rnn_dim, return_sequences=True, dropout_W=default_dropout_W, dropout_U=default_dropout_U)(rnn1forward)
            rnn1backward = LSTM(args.rnn_dim, return_sequences=True, dropout_W=default_dropout_W, dropout_U=default_dropout_U, go_backwards=True)(rnn1backward)
            rnn2forward = LSTM(args.rnn_dim, return_sequences=True, dropout_W=default_dropout_W, dropout_U=default_dropout_U)(rnn2forward)
            rnn2backward = LSTM(args.rnn_dim, return_sequences=True, dropout_W=default_dropout_W, dropout_U=default_dropout_U, go_backwards=True)(rnn2backward)
        
        if args.model_type == 'brnn':
            mergedHuman = merge([rnn1forward, rnn1backward], mode='concat', concat_axis=-1) # Concatenate the question and the answer by length (number of words)
            mergedChatbot = merge([rnn2forward, rnn2backward], mode='concat', concat_axis=-1) # Concatenate the question and the answer by length (number of words)
        elif args.model_type == 'brnn2':
            mergedHuman = merge([rnn1forward, rnn1backward], mode='concat', concat_axis=1) # Concatenate the question and the answer by length (number of words)
            mergedChatbot = merge([rnn2forward, rnn2backward], mode='concat', concat_axis=1) # Concatenate the question and the answer by length (number of words)
        
        merged = merge([mergedHuman, mergedChatbot], mode='concat', concat_axis=1) # Concatenate the question and the answer by length (number of words)

        # if pooling type is not specified, use attsum by default
        if not args.pooling_type:
            args.pooling_type = 'attsum'

        if args.pooling_type == 'meanot':
            pooled = MeanOverTime(mask_zero=True)(merged)
        elif args.pooling_type.startswith('att'):
            pooled = Attention(op=args.pooling_type, name="attention_layer")(merged)
        logger.info('%s pooling layer added!', args.pooling_type)

        densed = Dense(1)(pooled)
        score = Activation('sigmoid')(densed)
        model = Model(input=[sequence1, sequence2], output=score)
        
        # get the WordEmbedding layer index
        model.emb_index = []
        model_layer_index = 0
        for test in model.layers:
            if (test.name in {'Embedding1', 'Embedding2'}):
                model.emb_index.append(model_layer_index)
            model_layer_index += 1
    
    elif args.model_type == 'vdcnn':
        logger.info('Building a Very Deep CNN model')
        assert (args.cnn_dim > 0)

        model = Sequential()
        from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
        sequence1 = Input(shape=(None,), dtype='int32')
        sequence2 = Input(shape=(None,), dtype='int32')
        embed1 = Embedding(args.vocab_size, args.emb_dim, mask_zero=True, name="Embedding1")(sequence1)
        embed2 = Embedding(args.vocab_size, args.emb_dim, mask_zero=True, name="Embedding2")(sequence2)
        
        conv1 = embed1
        conv2 = embed2

        curr_nb_filter = args.cnn_dim
        conv1 = Conv1DWithMasking(nb_filter=curr_nb_filter*2, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1, activation='relu')(conv1)
        conv1 = Dropout(default_dropout)(conv1)
        conv2 = Conv1DWithMasking(nb_filter=curr_nb_filter*2, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1, activation='relu')(conv2)
        conv2 = Dropout(default_dropout)(conv2)

        for i in range(args.cnn_layer):
            curr_nb_filter = curr_nb_filter * 2
            conv1 = Conv1DWithMasking(nb_filter=curr_nb_filter, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1, activation='relu')(conv1)
            conv1 = Conv1DWithMasking(nb_filter=curr_nb_filter, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1, activation='relu')(conv1)
            conv1 = MaxPooling1DWithMasking()(conv1)
            conv1 = Dropout(default_dropout)(conv1)
            conv2 = Conv1DWithMasking(nb_filter=curr_nb_filter, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1, activation='relu')(conv2)
            conv2 = Conv1DWithMasking(nb_filter=curr_nb_filter, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1, activation='relu')(conv2)
            conv2 = MaxPooling1DWithMasking()(conv2)
            conv2 = Dropout(default_dropout)(conv2)

        merged = merge([conv1, conv2], mode='concat', concat_axis=1) # Concatenate the question and the answer by length (number of words)

        # if pooling type is not specified, use attsum by default
        if not args.pooling_type:
            args.pooling_type = 'attsum'

        if args.pooling_type == 'meanot':
            pooled = MeanOverTime(mask_zero=True)(merged)
        elif args.pooling_type.startswith('att'):
            pooled = Attention(op=args.pooling_type, name="attention_layer")(merged)
        logger.info('%s pooling layer added!', args.pooling_type)

        densed = Dense(1)(pooled)
        score = Activation('sigmoid')(densed)
        model = Model(input=[sequence1, sequence2], output=score)
        
        # get the WordEmbedding layer index
        model.emb_index = []
        model_layer_index = 0
        for test in model.layers:
            if (test.name in {'Embedding1', 'Embedding2'}):
                model.emb_index.append(model_layer_index)
            model_layer_index += 1

    else:
        raise NotImplementedError

    logger.info('Building model completed!')
    logger.info('  Done')
    
    ###############################################################################################################################
    ## Initialize embeddings if requested
    #

    if args.emb_path:
        from chatbotscorer.w2vEmbReader import W2VEmbReader as EmbReader
        logger.info("Loading embedding data...")
        global emb_reader
        if emb_reader == None:
            emb_reader = EmbReader(args.emb_path, emb_dim=args.emb_dim)
        if (isinstance(model.emb_index,list)):
            for emb_index in model.emb_index:
                model.layers[emb_index].W.set_value(emb_reader.get_emb_matrix_given_vocab(vocab, model.layers[emb_index].W.get_value()))
        else:
            model.layers[model.emb_index].W.set_value(emb_reader.get_emb_matrix_given_vocab(vocab, model.layers[model.emb_index].W.get_value()))
        logger.info("Loading embedding data completed!")

    return model

def load_model_architecture_and_weights(args):
    import keras.backend as K
    from keras.layers.embeddings import Embedding
    from keras.models import Sequential, Model, model_from_json
    from keras.layers.core import Dense, Dropout, Activation
    from chatbotscorer.my_layers import Attention, MeanOverTime, Conv1DWithMasking
    logger.info('Loading model architecture from: ' + args.arch_path)
    with open(args.arch_path, 'r') as arch_file:
        model = model_from_json(arch_file.read(), custom_objects={
            "MeanOverTime": MeanOverTime, "Attention": Attention, "Conv1DWithMasking":Conv1DWithMasking})
    logger.info('Loading model weights from: ' + args.model_weight_path)
    model.load_weights(args.model_weight_path)
    logger.info('Loading model architecture and weights completed!')

    return model

def load_complete_model(args):
    from keras.models import load_model
    logger.info('Loading complete model from: ' + args.model_path)
    model = load_model(args.model_path)
    logger.info('Loading complete model completed!')
    return model
