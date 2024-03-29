# Note: this class is serialized and passed through queues - so no methods
# are allowed as per pickle protocol

class Clip(object):
    def __init__(self):
        self.clip_id = None
        self.clip_path = None
        # format:
        # {fn: cv2_im, }
        self.framedb = None
        # format:
        # {fn: {
        #   nms_boxes: [[[ bbox_coords_plus_score ]], ] -> shape: (num_classes, (num_boxes, (blob))),
        #                                               -> blob: [x0, y0, x2, y2, scr, zdist]
        #   fc7_inds: [[], ] -> shape: (num_classes, (num_boxes)),
        #   fc7_scores: [[], ] -> shape: (num_uniq_boxes, (4096))
        # }, }
        self.predb = None
        # format:
        # {fn:
        #   conf_inds: [cls1ScThresh][cls2ScThresh][interThresh][cls1Idx][cls2Idx] => [idx_of_inter],
        #   conf_mat: [[ blob ]] -> shape: (num_classes, num_classes),
        #                        -> blob: [[ inter_score ]] -> shape (num_boxes_cls1, num_boxes_cls2)
        # }
        self.confdb = None
        self.result_path = None
