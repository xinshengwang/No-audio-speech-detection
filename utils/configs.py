class opts(object):
    # Acl RCNN parameter
    in_channel = 3
    hid_channel = [64,128]
    out_channel = 256
    kernel = [5,3,3]
    stride = [5,2,2]
    padding = [2,1,1]
    rnn_layer = 2
    rnn_hid_size = 256
    rnn_dropout = 0.5
    bidirectional = True

