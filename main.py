import argparse
import os
import src.config as config
import src.task as task


def parse_args():
    parser = argparse.ArgumentParser(description='Some model parameters')
    parser.add_argument('--json', '-j', type=str, help='use json config for model', default='none')
    parser.add_argument('--model', '-m', type=str, help='the name for the model used', default='basic_lstm')
    parser.add_argument('--learning_rate', '-lr', type=float, help='learning rate for each layer', default=0.0025)
    parser.add_argument('--lambda_l2', '-lb', type=float, help='lambda l2', default=0.0015)
    parser.add_argument('--epochs', '-ep', type=int, help='epochs for training', default=5)
    parser.add_argument('--batch_size', '-bc', type=int, help='batch size', default=256)
    parser.add_argument('--dropout_rate', '-dr', type=float, help='dropout rate', default=0.5)
    parser.add_argument('--hidden', '-hd', type=int, help='hidden size', default=32)
    parser.add_argument('--time_steps', '-ts', type=int, help='time steps', default=128)
    parser.add_argument('--clipping_threshold', '-cl', type=float, help='clip', default=15.0)
    parser.add_argument('--dataset_path', '-dp', type=str, help='dataset path', default="data/UCI/UCI HAR Dataset/")

    return parser.parse_args()

def read_json(path):
    import json 
    with open(path, 'r') as f:
        config = json.load(f)
    return config

if __name__ == '__main__':
    args = parse_args()

    labdir = 'lab1'
    if not os.path.exists(f'./{labdir}'):  # create figures and log folder if not exists 
        os.makedirs(f'./{labdir}')

    if args.json != 'none':
        print(f'config by {args.json}')
        json = read_json(args.json)
        conf = config.Config(
            model = json['model'],
            input_signal_types=json['input_signal_types'],
            labels=json['labels'],
            dataset_path = json['dataset_path'],
            time_steps = json['time_steps'],
            hidden = json['hidden'],
            batch_size = json['batch_size'],
            epochs = json['epochs'],
            learning_rate = json['learning_rate'],
            dropout_rate = json['dropout_rate'],
            lambda_l2 = json['lambda_l2'],
            clipping_threshold = json['clipping_threshold'])
    else:
        conf = config.Config(
            model = args.model,
            input_signal_types=['body_acc_x', 'body_acc_y', 'body_acc_z', 'body_gyro_x', 'body_gyro_y', 'body_gyro_z', 'total_acc_x', 'total_acc_y', 'total_acc_z'],
            labels=['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING'],
            dataset_path = args.dataset_path,
            time_steps = args.time_steps,
            hidden = args.hidden,
            batch_size = args.batch_size,
            epochs = args.epochs,
            learning_rate = args.learning_rate,
            dropout_rate = args.dropout_rate,
            lambda_l2 = args.lambda_l2,
            clipping_threshold = args.clipping_threshold)
    
    print(conf)

    t = task.Task(conf)
    t.figure(t.evaluate(t.train()), plot=False)
