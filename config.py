import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Image Classification Fire')

    """
    General configuration
    """
    parser.add_argument('--cuda', default=True, help='Weather use cuda.')
    parser.add_argument('--num_classes', default=2, help='The number of the classes')
    parser.add_argument('--class_names', default= ['no_fire', 'fire'])
    parser.add_argument('--backbone', default='resnet18', help='The backbone network that will be used')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate.')
    parser.add_argument('--batch_size', default=8, help='The batch size used by a single GPU during training')
    parser.add_argument('--max_epoch', default=10, help='The maximum epoch used in this training')
    parser.add_argument('--pretrained', default=True, help='Whether to use pre-training weights')

    return parser.parse_args()