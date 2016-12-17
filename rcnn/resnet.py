import mxnet as mx
import rpn.proposal, rpn.proposal_target
from config import config

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=512,
                  bn_global=True):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=bn_global,
                               name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=bn_global,
                               name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=bn_global
                               , name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, use_global_stats=bn_global
                               , name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, use_global_stats=bn_global
                               , name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        return conv2 + shortcut


def rpn(data, num_class=2, num_anchor=12, is_train=False):
    """Return RPN+ROIPooling Unit
    Parameters
    ----------
    data : str
        Input data
    num_anchors : int
        Number of anchors
    num_classes : int
        number class of your detction task(include the background)
    """
    im_info = mx.symbol.Variable(name="im_info")

    if is_train:
        label = mx.symbol.Variable(name='label')
        bbox_target = mx.symbol.Variable(name='bbox_target')
        bbox_inside_weight = mx.symbol.Variable(name='bbox_inside_weight')
        bbox_outside_weight = mx.symbol.Variable(name='bbox_outside_weight')
        gt_boxes = mx.symbol.Variable(name="gt_boxes")
        gt_boxes = mx.symbol.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')

    rpn_conv = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchor, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchor, name="rpn_bbox_pred")
    rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

    if is_train:
        # cls
        rpn_cls_loss = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=label, multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_loss")
        # reg
        rpn_bbox_loss_ = bbox_outside_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=bbox_inside_weight * (rpn_bbox_pred - bbox_target))
        rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_)

    rpn_cls_prob = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
    rpn_cls_prob_reshape = mx.symbol.Reshape(data=rpn_cls_prob, shape=(0, 2 * num_anchor, -1, 0), name='rpn_cls_prob_reshape')
    if is_train:
        rpn_roi = mx.symbol.Custom(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rpn_rois',
        op_type='proposal', feat_stride=16, scales=(4, 8, 16, 32), ratios=(0.5, 1, 2), is_train=True)
        rois = mx.symbol.Custom(
            rpn_roi=rpn_roi, gt_boxes=gt_boxes, name='rois', op_type='proposal_target',
            num_classes=num_class, is_train=True)
        roi_pool = mx.symbol.ROIPooling(name='roi_pool5', data=data, rois=rois[0], pooled_size=(7, 7), spatial_scale=0.0625)
        return roi_pool, rois, rpn_cls_loss, rpn_bbox_loss
    else:
        rpn_roi = mx.symbol.Custom(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rpn_rois',
        op_type='proposal', feat_stride=16, scales=(4, 8, 16, 32), ratios=(0.5, 1, 2), is_train=False)
        roi_pool = mx.symbol.ROIPooling(name='roi_pool5', data=data, rois=rpn_roi, pooled_size=(7, 7), spatial_scale=0.0625)
        return roi_pool, rpn_roi


def resnet(units, num_stage, filter_list, num_class=2, num_anchor=12, bottle_neck=True, bn_mom=0.9,
           bn_global=True, workspace=512, is_train=False):
    """Return ResNet symbol of cifar10 and imagenet
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stage : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_class : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    rois = None
    rpn_cls_loss = None
    rpn_bbox_loss = None
    num_unit = len(units)
    assert(num_unit == num_stage)
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, use_global_stats=bn_global, name='bn_data')
    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                              no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=bn_global, name='bn0')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
    body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    for i in range(num_stage):
        bn_global_ = bn_global if i < num_stage-1 else False  # after roi-pooling, do not use use_global_stats
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             bn_global=bn_global_)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, bn_global=bn_global_)
        if i == num_stage - 2:
            # put RPN and ROI Pooling here, i.e.the last of stage 3
            if is_train:
                body, rois, rpn_cls_loss, rpn_bbox_loss = rpn(body, num_class=num_class, num_anchor=num_anchor, is_train=True)
            else:
                body, rpn_roi = rpn(body, num_class=num_class, num_anchor=num_anchor, is_train=False)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=False, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)

    # cls
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=flat, num_hidden=num_class)
    if is_train:
        cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=rois[1], normalization='batch')
    else:
        cls_prob = mx.symbol.SoftmaxActivation(name='cls_prob', data=cls_score)

    # reg
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=flat, num_hidden=num_class * 4)
    if is_train:
        bbox_loss_ = rois[4] * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=rois[3] * (bbox_pred - rois[2]))
        bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_SIZE)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TRAIN.IMS_PER_BATCH, -1, num_class), name='cls_prob_reshape')
    if is_train:
        bbox_pred = mx.symbol.Reshape(data=bbox_loss, shape=(config.TRAIN.IMS_PER_BATCH, -1, 4 * num_class), name='bbox_pred_reshape')
    else:
        bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TRAIN.IMS_PER_BATCH, -1, 4 * num_class), name='bbox_pred_reshape')

    if is_train:
        return mx.symbol.Group([rois[1], rpn_cls_loss, rpn_bbox_loss, cls_prob, bbox_pred])
    else:
        return mx.symbol.Group([rpn_roi, cls_prob, bbox_pred])


def resnet_18(num_class=2, bn_mom=0.99, bn_global=True, is_train=False):
    return resnet(units=[2, 2, 2, 2], num_stage=4, filter_list=[64, 64, 128, 256, 512], num_class=num_class,
                  num_anchor=12, bottle_neck=False, bn_mom=bn_mom, bn_global=bn_global, workspace=512, is_train=is_train)


def resnet_34(num_class=2, bn_mom=0.99, bn_global=True, is_train=False):
    return resnet(units=[3, 4, 6, 3], num_stage=4, filter_list=[64, 64, 128, 256, 512], num_class=num_class,
                  num_anchor=12, bottle_neck=False, bn_mom=bn_mom, bn_global=bn_global, workspace=512, is_train=is_train)


def resnet_50(num_class=2, bn_mom=0.99, bn_global=True, is_train=False):
    return resnet(units=[3, 4, 6, 3], num_stage=4, filter_list=[64, 256, 512, 1024, 2048], num_class=num_class,
                  num_anchor=12, bottle_neck=True, bn_mom=bn_mom, bn_global=bn_global, workspace=512, is_train=is_train)


def resnet_101(num_class=2, bn_mom=0.99, bn_global=True, is_train=False):
    return resnet(units=[3, 4, 23, 3], num_stage=4, filter_list=[64, 256, 512, 1024, 2048], num_class=num_class,
                  num_anchor=12, bottle_neck=True, bn_mom=bn_mom, bn_global=bn_global, workspace=512, is_train=is_train)


def resnet_152(num_class=2, bn_mom=0.99, bn_global=True, is_train=False):
    return resnet(units=[3, 8, 36, 3], num_stage=4, filter_list=[64, 256, 512, 1024, 2048], num_class=num_class,
                  num_anchor=12, bottle_neck=True, bn_mom=bn_mom, bn_global=bn_global, workspace=512, is_train=is_train)


def resnet_200(num_class=2, bn_mom=0.99, bn_global=True, is_train=False):
    return resnet(units=[3, 24, 36, 3], num_stage=4, filter_list=[64, 256, 512, 1024, 2048], num_class=num_class,
                  num_anchor=12, bottle_neck=True, bn_mom=bn_mom, bn_global=bn_global, workspace=512, is_train=is_train)
