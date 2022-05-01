import os
from tqdm.auto import tqdm
from typing import List
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from . import text_regression_pipeline, utils

HUGGINGFACE_BASE_MODEL = "dccuchile/bert-base-spanish-wwm-uncased"

transformer_models_by_tag = {
    'raw_label': {
        'regression': False,
        'target_file': utils.RAW_PREPROC_DATA,
        'target_column': 'label',
    }, 
    'regression_w_m_vote': {
        'regression': True,
        'target_file': utils.AGGREGATED_DATA,
        'target_column': 'w_m_vote',
    },
    'm_vote_strict': {
        'regression': False,
        'target_file': utils.AGGREGATED_DATA,
        'target_column': 'm_vote',
        'target_column_args': {
            'strict': True,
        }
    },
    'm_vote_nonstrict': {
        'regression': False,
        'target_file': utils.AGGREGATED_DATA,
        'target_column': 'm_vote',
        'target_column_args': {
            'strict': False,
        }
    },
    'w_m_vote_strict': {
        'regression': False,
        'target_file': utils.AGGREGATED_DATA,
        'target_column': 'w_m_vote',
        'target_column_args': {
            'strict': True,
        }
    },
    'w_m_vote_nonstrict': {
        'regression': False,
        'target_file': utils.AGGREGATED_DATA,
        'target_column': 'w_m_vote',
        'target_column_args': {
            'thr': 0.35
        }
    }
}

def load_transformer_model_from_path(model_path: str, regression=False):
    """
    Loads a transformers model from a specified location (local or from huggingface)
    model_path: str. Location of the model or HuggingFace path (e.g. "dccuchile/bert-base-spanish-wwm-uncased")
    regression: bool. Specifies whether the model is a regression model (True) or is a classification model (False)
    """
    if regression:
        num_labels = 1
        # TODO id2label? is it necessary?
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    else:
        num_labels = 2
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, id2label={0: 'non-racist', 1: 'racist'})
    return model


def load_tokenizer(model_name=HUGGINGFACE_BASE_MODEL):
    """
    Loads a transformers tokenizer
    model_name: str. Location of the model or huggingface path. Default to "dccuchile/bert-base-spanish-wwm-uncased"
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def load_transformer_model_from_tag(tag: str, epoch: int):
    """
    Loads a transformers model from a specified tag
    tag: str. One of the keys of `transformer_models_by_tag`: `raw_label`, `regression_w_m_vote`, `m_vote_strict`, `m_vote_nonstrict`, `w_m_vote_strict`, `w_m_vote_nonstrict`
    epoch: int. Which epoch to load, possible values: 1, 2, 3, 4

    This function loads the models from HuggingFace and stores them in the `~/.cache` folder.
    """
    if tag not in transformer_models_by_tag:
        raise ValueError("Model tag {} not found".format(tag))
    cfg = transformer_models_by_tag[tag]
    regression = cfg['regression']
    # model_path = f'models/{tag}/epoch_{epoch}'
    model_path = f'blind-reviews/racism-models-{tag.replace("_", "-")}-epoch-{epoch}'
    model = load_transformer_model_from_path(model_path, regression)
    return model


def load_pipeline_from_tag(tag: str, epoch: int):
    """
    Loads a `transformers.pipeline` from a specified tag
    tag: str. one of the keys of `transformer_models_by_tag`: `raw_label`, `regression_w_m_vote`, `m_vote_strict`, `m_vote_nonstrict`, `w_m_vote_strict`, `w_m_vote_nonstrict`
    epoch: int. Which epoch to load, possible values: 1, 2, 3, 4
    """
    if tag not in transformer_models_by_tag:
        raise ValueError("Model tag {} not found".format(tag))
    cfg = transformer_models_by_tag[tag]
    model = load_transformer_model_from_tag(tag, epoch)
    tokenizer = load_tokenizer()
    regression = cfg['regression']
    if not regression:
        pipe = pipeline("text-classification", model = model, tokenizer = tokenizer)
    else:
        pipe = text_regression_pipeline.TextRegressionPipeline(model=model, tokenizer=tokenizer)
    return pipe

def predict(pipe, texts: List[str], regression=False) -> List[str]:
    labels = []
    for out in pipe(texts):
        # print(out)
        if regression:
            labels.append(out['score'])
        else:
            labels.append(out['label'])
    return labels

def evaluate_pipe(tag, epoch, force_recomputation=False):
    """
    Evaluates a pipeline from a specified tag and epoch.
    This method saves the predictions for the validation and evaluation samples in the `data/predictions` folder.
    tag: str. The tag of the pipeline to evaluate
    epoch: int. The epoch to evaluate
    force_recomputation: bool. Whether to force the recomputation of the predictions or to use the previously saved ones.
    """
    output_path_validation = f'{utils.DATA_DIR}/predictions/validation_sample_{tag}_epoch_{epoch}.csv'
    output_path_evaluation = f'{utils.DATA_DIR}/predictions/evaluation_sample_{tag}_epoch_{epoch}.csv'
    if not os.path.exists(f'{utils.DATA_DIR}/predictions'):
        os.makedirs(f'{utils.DATA_DIR}/predictions')
    config = transformer_models_by_tag[tag]
    regression = config['regression']
    validation_set = utils.get_validation_set(config['target_file'])
    y_true_column = config['target_column']
    if 'target_column_args' in config:
        y_true_column_args = config['target_column_args']
        # print(validation_set)
        # print(y_true_column)
        # print(y_true_column_args)
        validation_y_true = utils.binarize_label(validation_set, col=y_true_column, **y_true_column_args)
    else:
        validation_y_true = validation_set[y_true_column]
    # and the evaluation set
    evaluation_set = utils.load_dataset(utils.EVAL_SAMPLE)
    #
    recompute_validation = force_recomputation or not os.path.exists(output_path_validation)
    recompute_evaluation = force_recomputation or not os.path.exists(output_path_evaluation)
    if recompute_validation or recompute_evaluation:
        print(f'[tag:{tag} epoch:{epoch}] Loading pipeline...')
        pipe = load_pipeline_from_tag(tag, epoch)
        print(f'[tag:{tag} epoch:{epoch}] Pipeline loaded!')
    if not recompute_validation:
        # print(f'[tag:{tag} epoch:{epoch}] Using previously computed validation set predictions')
        validation_predictions_table = utils.load_dataset(output_path_validation)
        if not regression:
            predictions_validation = validation_predictions_table['predicted_label']
        else:
            predictions_validation = validation_predictions_table['predicted_score']
    else:
        print(f'[tag:{tag} epoch:{epoch}] Predicting validation set...')
        predictions_validation = predict(pipe, validation_set['message'].tolist(), regression=regression)
        print(f'[tag:{tag} epoch:{epoch}] Validation set predicted!')
    #
    if not recompute_evaluation:
        # print(f'[tag:{tag} epoch:{epoch}] Using previously computed evaluation set predictions')
        evaluation_predictions_table = utils.load_dataset(output_path_evaluation)
        if not regression:
            predictions_evaluation = evaluation_predictions_table['predicted_label']
        else:
            predictions_evaluation = evaluation_predictions_table['predicted_score']
    else:
        print(f'[tag:{tag} epoch:{epoch}] Predicting evaluation set...')
        predictions_evaluation = predict(pipe, evaluation_set['message'].tolist(), regression=regression)
        print(f'[tag:{tag} epoch:{epoch}] Evaluation set predicted!')
    if not regression:
        f1_validation = utils.f1(validation_y_true, predictions_validation)
        f1_evaluation = utils.f1(evaluation_set['label'], predictions_evaluation)
        f1_average = (f1_validation + f1_evaluation) / 2
        # insert predicted_label column
        validation_set['predicted_label'] = predictions_validation
        evaluation_set['predicted_label'] = predictions_evaluation
        result = {'f1_validation': f1_validation, 'f1_evaluation': f1_evaluation, 'f1_average': f1_average}
    else:
        # insert predicted_label column (float)
        validation_set['predicted_score'] = predictions_validation
        evaluation_set['predicted_score'] = predictions_evaluation
        result = {}
        # and now determine the different predicted_label_0.1, predicted_label_0.15, ...
        score_to_label = lambda score, thr: 'racist' if score > thr else 'non-racist'
        for t in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
            validation_y_true_labels = [score_to_label(score, t) for score in validation_y_true]
            validation_y_pred_labels = [score_to_label(score, t) for score in predictions_validation]
            f1_validation = utils.f1(validation_y_true_labels, validation_y_pred_labels)
            evaluation_y_true_labels = evaluation_set['label']
            evaluation_y_pred_labels = [score_to_label(score, t) for score in predictions_evaluation]
            f1_evaluation = utils.f1(evaluation_y_true_labels, evaluation_y_pred_labels)
            f1_average = (f1_validation + f1_evaluation) / 2
            # save to the results
            result[t] = {'f1_validation': f1_validation, 'f1_evaluation': f1_evaluation, 'f1_average': f1_average}
            # add to the dataframe
            validation_set[f'predicted_label_{t}'] = validation_y_pred_labels
            evaluation_set[f'predicted_label_{t}'] = evaluation_y_pred_labels
    # and write to file if necessary
    if recompute_validation:
        validation_set.to_csv(output_path_validation, sep='|', index=False)
    if recompute_evaluation:
        evaluation_set.to_csv(output_path_evaluation, sep='|', index=False)
    return result

