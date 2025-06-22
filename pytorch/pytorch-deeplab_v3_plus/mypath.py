class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/data/datasets/VOCdevkit/VOC2012/'  
        elif dataset == 'sbd':
            return '/data/datasets/benchmark_RELEASE/'  
        elif dataset == 'cityscapes':
            return '/data/datasets/cityscapes/'     
        elif dataset == 'coco':
            return '/data/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
