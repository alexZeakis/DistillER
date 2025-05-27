import logging
import os
from datetime import datetime
import json

from supervised_utils import initialize_gpu_seed, load_data, DataType, \
                             DeepMatcherProcessor, predict,  setup_logging, \
                             read_arguments_train, load_model
from time import time

setup_logging()


def create_experiment_folder(model_output_dir: str, model_type: str, data_name: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_name = "{}__{}__{}".format(data_name.upper(), model_type.upper(), timestamp)

    output_path = os.path.join(model_output_dir, experiment_name)
    os.makedirs(output_path, exist_ok=True)

    return experiment_name


if __name__ == "__main__":
    args = read_arguments_train()  # same arg parser used in training

    model_dir = args.model_name_or_path
    device, n_gpu = initialize_gpu_seed(args.device, args.seed)

    processor = DeepMatcherProcessor()
    label_list = processor.get_labels()

    model, tokenizer = load_model(model_dir, device)
    logging.info("Model loaded from {}".format(model_dir))

    test_examples = processor.get_test_examples(args.data_path)
    logging.info("Loaded {} test examples".format(len(test_examples)))

    test_data_loader = load_data(
        test_examples,
        label_list,
        tokenizer,
        args.eval_batch_size,
        DataType.TEST,
        args.model_type
    )

    include_token_type_ids = args.model_type == 'bert'

    t1 = time()
    simple_accuracy, f1, classification_report, prfs, predictions = predict(
        model, device, test_data_loader, include_token_type_ids
    )
    t2 = time()
    testing_time = t2 - t1

    logging.info("Prediction done. F1: {}, Accuracy: {}".format(f1, simple_accuracy))
    logging.info(classification_report)

    # Save predictions
    path2 = os.path.join(args.log_dir, args.data_name + '_predictions.csv')
    os.makedirs(os.path.dirname(path2), exist_ok=True)
    predictions.to_csv(path2)

    # Save evaluation log
    keys = ['precision', 'recall', 'fbeta_score', 'support']
    prfs = {f'class_{no}': {key: float(prfs[nok][no]) for nok, key in enumerate(keys)} for no in range(2)}

    log_file = os.path.join(args.log_dir, 'matching_supervised_dynamic.txt')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a') as fout:
        scores = {
            'simple_accuracy': simple_accuracy,
            'f1': f1,
            'model_type': args.model_type,
            'data_name': args.data_name,
            'testing_time': testing_time,
            'prfs': prfs
        }
        fout.write(json.dumps(scores) + "\n")