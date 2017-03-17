import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_model(args, overal_maxlen, vocab, emb_reader):

    import keras.backend as K
    from keras.layers.embeddings import Embedding
    from keras.models import Sequential, Model
    from keras.layers.core import Dense, Dropout, Activation
    from medlstm.my_layers import Attention, MeanOverTime, Conv1DWithMasking

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
    
    if args.model_type == 'cls':
        raise NotImplementedError

    elif args.model_type == 'rnn':
        logger.info('Building a RNN model')
        assert (args.rnn_dim > 0)

        model = Sequential()
        model.add(Embedding(args.vocab_size, args.emb_dim, mask_zero=True))

        for i in range(args.rnn_layer - 1):
            model.add(RNN(args.rnn_dim, dropout_W=default_dropout_W, dropout_U=default_dropout_U, return_sequences=True))
            model.add(Dropout(default_dropout)) # Needed between two LSTM layers

        # if pooling type is specified
        if args.pooling_type:
            model.add(RNN(args.rnn_dim, return_sequences=True, dropout_W=default_dropout_W, dropout_U=default_dropout_U))  # return_sequences true
            model.add(Dropout(default_dropout))
            if args.pooling_type == 'meanot':
                model.add(MeanOverTime(mask_zero=True))
            elif args.pooling_type.startswith('att'):
                model.add(Attention(op=args.pooling_type, name="attention_layer"))
            logger.info('%s pooling layer added!', args.pooling_type)
        else:
            model.add(RNN(args.rnn_dim, dropout_W=default_dropout_W, dropout_U=default_dropout_U))  # return_sequences false
            model.add(Dropout(default_dropout))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.emb_index = 0

    elif args.model_type == 'brnn':
        logger.info('Building a BIDIRECTIONAL RNN model')
        assert (args.rnn_dim > 0)

        model = Sequential()
        from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
        sequence = Input(shape=(None,), dtype='int32')
        output = Embedding(args.vocab_size, args.emb_dim, mask_zero=True)(sequence)
        # if there is pooling layer, return sequences
        if args.pooling_type:
            forwards = LSTM(args.rnn_dim, return_sequences=True, dropout_W=default_dropout_W, dropout_U=default_dropout_U)(output)
            backwards = LSTM(args.rnn_dim, return_sequences=True, dropout_W=default_dropout_W, dropout_U=default_dropout_U, go_backwards=True)(output)
        else:
            forwards = LSTM(args.rnn_dim, return_sequences=False, dropout_W=default_dropout_W, dropout_U=default_dropout_U)(output)
            backwards = LSTM(args.rnn_dim, return_sequences=False, dropout_W=default_dropout_W, dropout_U=default_dropout_U, go_backwards=True)(output)
        forwards_dp = Dropout(default_dropout)(forwards)
        backwards_dp = Dropout(default_dropout)(backwards)
        merged = merge([forwards_dp, backwards_dp], mode='concat', concat_axis=-1)
        
        # If pooling layer specified, add pooling
        if args.pooling_type:
            if args.pooling_type == 'meanot':
                merged = MeanOverTime(mask_zero=True)(merged)
            elif args.pooling_type.startswith('att'):
                merged = Attention(op=args.pooling_type, name="attention_layer")(merged)
            logger.info('%s pooling layer added!', args.pooling_type)

        densed = Dense(1)(merged)
        score = Activation('sigmoid')(densed)
        model = Model(input=sequence, output=score)
        model.emb_index = 1

    elif args.model_type == 'cnn':
        logger.info('Building a CNN model')
        assert (args.cnn_dim > 0)

        model = Sequential()
        model.add(Embedding(args.vocab_size, args.emb_dim, mask_zero=True))

        for i in range(args.cnn_layer):
            model.add(Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1))
            model.add(Dropout(default_dropout))

        # if pooling type is not specified, use attsum by default
        if not args.pooling_type:
            args.pooling_type = 'attsum'

        if args.pooling_type == 'meanot':
            model.add(MeanOverTime(mask_zero=True))
        elif args.pooling_type.startswith('att'):
            model.add(Attention(op=args.pooling_type, name="attention_layer"))
        logger.info('%s pooling layer added!', args.pooling_type)

        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.emb_index = 0

    elif args.model_type == 'rcnn':
        logger.info('Building a RCNN model')
        assert (args.cnn_dim > 0)
        assert (args.rnn_dim > 0)

        model = Sequential()
        model.add(Embedding(args.vocab_size, args.emb_dim, mask_zero=True))

        # RNN layer
        for i in range(args.rnn_layer):
            model.add(RNN(args.rnn_dim, dropout_W=default_dropout_W, dropout_U=default_dropout_U, return_sequences=True))
            model.add(Dropout(default_dropout)) # Needed between two LSTM layers
        
        for i in range(args.cnn_layer):
            model.add(Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1))
            model.add(Dropout(default_dropout))

       # if pooling type is not specified, use attsum by default
        if not args.pooling_type:
            args.pooling_type = 'attsum'

        if args.pooling_type == 'meanot':
            model.add(MeanOverTime(mask_zero=True))
        elif args.pooling_type.startswith('att'):
            model.add(Attention(op=args.pooling_type, name="attention_layer"))
        logger.info('%s pooling layer added!', args.pooling_type)

        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.emb_index = 0

    elif args.model_type == 'crnn':
        logger.info('Building a CRNN model')
        assert (args.cnn_dim > 0)
        assert (args.rnn_dim > 0)

        model = Sequential()
        model.add(Embedding(args.vocab_size, args.emb_dim, mask_zero=True))

        for i in range(args.cnn_layer):
            model.add(Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1))
            model.add(Dropout(default_dropout))

        for i in range(args.rnn_layer - 1):
            model.add(RNN(args.rnn_dim, dropout_W=default_dropout_W, dropout_U=default_dropout_U, return_sequences=True))
            model.add(Dropout(default_dropout)) # Needed between two LSTM layers

        # if pooling type is specified
        if args.pooling_type:
            model.add(RNN(args.rnn_dim, return_sequences=True, dropout_W=default_dropout_W, dropout_U=default_dropout_U))  # return_sequences true
            model.add(Dropout(default_dropout))
            if args.pooling_type == 'meanot':
                model.add(MeanOverTime(mask_zero=True))
            elif args.pooling_type.startswith('att'):
                model.add(Attention(op=args.pooling_type, name="attention_layer"))
            logger.info('%s pooling layer added!', args.pooling_type)
        else:
            model.add(RNN(args.rnn_dim, dropout_W=default_dropout_W, dropout_U=default_dropout_U))  # return_sequences false
            model.add(Dropout(default_dropout))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.emb_index = 0
        
    elif args.model_type == 'cwrnn':
        logger.info('Building a CWRNN model (CNN then concat with word embeddings, then passed to RNN)')
        assert (args.cnn_dim > 0)
        assert (args.rnn_dim > 0)
        assert (args.cnn_layer > 0)
        assert (args.rnn_layer > 0)

        model = Sequential()
        from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
        sequence = Input(shape=(None,), dtype='int32')
        embed = Embedding(args.vocab_size, args.emb_dim, mask_zero=True)(sequence)
        conv = embed

        for i in range(args.cnn_layer):
            conv = Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1)(conv)
            conv = Dropout(default_dropout)(conv)
        
        convMergeWord = merge([embed,conv], mode='concat', concat_axis=-1)
        recc = convMergeWord

        for i in range(args.rnn_layer - 1):
            recc = RNN(args.rnn_dim, dropout_W=default_dropout_W, dropout_U=default_dropout_U, return_sequences=True)(recc)
            recc = Dropout(default_dropout)(recc)

        # if pooling type is specified
        if args.pooling_type:
            recc = RNN(args.rnn_dim, return_sequences=True, dropout_W=default_dropout_W, dropout_U=default_dropout_U)(recc)
            recc = Dropout(default_dropout)(recc)
            if args.pooling_type == 'meanot':
                recc = MeanOverTime(mask_zero=True)(recc)
            elif args.pooling_type.startswith('att'):
                recc = Attention(op=args.pooling_type, name="attention_layer")(recc)
            logger.info('%s pooling layer added!', args.pooling_type)
        else:
            recc = RNN(args.rnn_dim, dropout_W=default_dropout_W, dropout_U=default_dropout_U)(recc)
            recc = Dropout(default_dropout)(recc)

        densed = Dense(1)(recc)
        score = Activation('sigmoid')(densed)
        model = Model(input=sequence, output=score)
        model.emb_index = 1
        
    elif args.model_type == 'cwbrnn':
        logger.info('Building a CWBRNN model (CNN then concat with word embeddings, then passed to Bidirectional RNN)')
        assert (args.cnn_dim > 0)
        assert (args.rnn_dim > 0)
        assert (args.cnn_layer > 0)
        assert (args.rnn_layer > 0)

        model = Sequential()
        from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
        sequence = Input(shape=(None,), dtype='int32')
        embed = Embedding(args.vocab_size, args.emb_dim, mask_zero=True)(sequence)
        conv = embed

        for i in range(args.cnn_layer):
            conv = Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1)(conv)
            conv = Dropout(default_dropout)(conv)
        
        convMergeWord = merge([embed,conv], mode='concat', concat_axis=-1)

        if args.pooling_type:
            forwards = LSTM(args.rnn_dim, return_sequences=True, dropout_W=default_dropout_W, dropout_U=default_dropout_U)(convMergeWord)
            backwards = LSTM(args.rnn_dim, return_sequences=True, dropout_W=default_dropout_W, dropout_U=default_dropout_U, go_backwards=True)(convMergeWord)
        else:
            forwards = LSTM(args.rnn_dim, return_sequences=False, dropout_W=default_dropout_W, dropout_U=default_dropout_U)(convMergeWord)
            backwards = LSTM(args.rnn_dim, return_sequences=False, dropout_W=default_dropout_W, dropout_U=default_dropout_U, go_backwards=True)(convMergeWord)
        forwards_dp = Dropout(default_dropout)(forwards)
        backwards_dp = Dropout(default_dropout)(backwards)
        merged = merge([forwards_dp, backwards_dp], mode='concat', concat_axis=-1)
        
        # If pooling layer specified, add pooling
        if args.pooling_type:
            if args.pooling_type == 'meanot':
                merged = MeanOverTime(mask_zero=True)(merged)
            elif args.pooling_type.startswith('att'):
                merged = Attention(op=args.pooling_type, name="attention_layer")(merged)
            logger.info('%s pooling layer added!', args.pooling_type)

        densed = Dense(1)(merged)
        score = Activation('sigmoid')(densed)
        model = Model(input=sequence, output=score)
        model.emb_index = 1

    elif args.model_type == 'cnn+cnn':
        logger.info('Building a CNN + CNN model (independent CNN, then concatenated)')
        assert (args.cnn_dim > 0)

        model = Sequential()
        from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
        sequence = Input(shape=(None,), dtype='int32')
        embed = Embedding(args.vocab_size, args.emb_dim, mask_zero=True)(sequence)

        conv = [None] * args.cnn_layer
        for i in range(args.cnn_layer):
            conv[i] = Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size+i, border_mode=cnn_border_mode, subsample_length=1)(embed)
            conv[i] = Dropout(default_dropout)(conv[i])
        if args.cnn_layer > 1:
            merged_conv = merge(conv, mode='concat', concat_axis=-1)
        else:
            merged_conv = conv[0]
        merged = merged_conv

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
        model = Model(input=sequence, output=score)
        model.emb_index = 1

    elif args.model_type == 'rnn+cnn':
        logger.info('Building a RNN + CNN model (independent RNN and CNN, then concatenated)')
        assert (args.cnn_dim > 0)
        assert (args.rnn_dim > 0)

        model = Sequential()
        from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
        sequence = Input(shape=(None,), dtype='int32')
        embed = Embedding(args.vocab_size, args.emb_dim, mask_zero=True)(sequence)

        conv = [None] * args.cnn_layer
        for i in range(args.cnn_layer):
            conv[i] = Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size+i, border_mode=cnn_border_mode, subsample_length=1)(embed)
            conv[i] = Dropout(default_dropout)(conv[i])
        if args.cnn_layer > 1:
            merged_conv = merge(conv, mode='concat', concat_axis=-1)
        else:
            merged_conv = conv[0]

        for i in range(args.rnn_layer):
            recurrent = LSTM(args.rnn_dim, dropout_W=default_dropout_W, dropout_U=default_dropout_U, return_sequences=True)(embed)
            recurrent = Dropout(default_dropout)(recurrent)

        merged = merge([merged_conv, recurrent], mode='concat', concat_axis=-1)

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
        model = Model(input=sequence, output=score)
        model.emb_index = 1

    elif args.model_type == 'bclslab':
        logger.info('Building a BINARY CLASSIFICATION model with TWO LAB RESULTS (Real numbers)')
        model = Sequential()
        from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
        # default sequence (sequence of word indexes)
        sequence = Input(shape=(overal_maxlen,), dtype='int32')
        # two lab results input layer
        sequenceLab = Input(shape=(2,), dtype='float32')
        output = Embedding(args.vocab_size, args.emb_dim, mask_zero=True)(sequence)
        forward_layer= (output)
        for i in range(args.rnn_layer - 1):
            forward_layer = LSTM(args.rnn_dim, dropout_W=default_dropout_W, dropout_U=default_dropout_U, return_sequences=True)(forward_layer)
            forward_layer = Dropout(default_dropout)(forward_layer) # Needed between two LSTM layers
        forwards = LSTM(args.rnn_dim, return_sequences=False, dropout_W=default_dropout_W, dropout_U=default_dropout_U)(forward_layer)
        forwards_dp = Dropout(default_dropout)(forwards) # with 0.5 dropout
        merged = merge([forwards_dp, sequenceLab], mode='concat', concat_axis=-1)
        densed = Dense(1)(merged)
        score = Activation('sigmoid')(densed)
        model = Model(input=[sequence, sequenceLab], output=score)
        model.emb_index = 1
    
    else:
        raise NotImplementedError

    logger.info('Building model completed!')
    logger.info('  Done')
    
    ###############################################################################################################################
    ## Initialize embeddings if requested
    #

    if emb_reader:
        from medlstm.w2vEmbReader import W2VEmbReader as EmbReader
        logger.info('Initializing lookup table...')
        model.layers[model.emb_index].W.set_value(emb_reader.get_emb_matrix_given_vocab(vocab, model.layers[model.emb_index].W.get_value()))
        logger.info('Initializing lookup table completed!')
    return model

def load_embedding_reader(args):
    import pickle as pk
    emb_reader = None
    if args.emb_binary_path:
        from medlstm.w2vEmbReader import W2VEmbReader as EmbReader
        logger.info("Loading binary embedding data...")
        with open(args.emb_binary_path, 'rb') as emb_data_file:
            emb_reader = pk.load(emb_data_file)
        logger.info("Loading binary embedding data completed!")
    else:
        if args.emb_path:
            from medlstm.w2vEmbReader import W2VEmbReader as EmbReader
            logger.info("Loading embedding data...")
            emb_reader = EmbReader(args.emb_path, emb_dim=args.emb_dim)
            with open(args.out_dir_path + '/data/emb_reader_instance_'+ str(args.emb_dim) +'.pkl', 'wb') as emb_data_file:
                pk.dump(emb_reader, emb_data_file)
            logger.info("Loading embedding data completed!")
    return emb_reader

def load_model_architecture_and_weights(args):
    import keras.backend as K
    from keras.layers.embeddings import Embedding
    from keras.models import Sequential, Model, model_from_json
    from keras.layers.core import Dense, Dropout, Activation
    from medlstm.my_layers import Attention, MeanOverTime, Conv1DWithMasking
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
