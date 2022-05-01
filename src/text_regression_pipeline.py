from typing import Dict
import numpy as np
from transformers.pipelines import TextClassificationPipeline


class TextRegressionPipeline(TextClassificationPipeline):
    """
    Class based on the TextClassificationPipeline from transformers.
    The difference is that instead of being based on a classifier, it is based on a regressor.
    You can specify the regression threshold when you call the pipeline or when you instantiate the pipeline.
    """

    def __init__(self, **kwargs):
        """
        Builds a new Pipeline based on regression.
        regression_threshold: Optional(float). If None, the pipeline will simply output the score. If set to a specific value, the output will be both the score and the label.
        """
        self.regression_threshold = kwargs.pop("regression_threshold", None)
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        """
        You can also specify the regression threshold when you call the pipeline.
        regression_threshold: Optional(float). If None, the pipeline will simply output the score. If set to a specific value, the output will be both the score and the label.
        """
        self.regression_threshold_call = kwargs.pop("regression_threshold", None)
        result = super().__call__(*args, **kwargs)
        return result


    def postprocess(self, model_outputs, function_to_apply=None, return_all_scores=False):
        outputs = model_outputs["logits"][0]
        outputs = outputs.numpy()

        scores = outputs
        score = scores[0]
        regression_threshold = self.regression_threshold
        # override the specific threshold if it is specified in the call
        if self.regression_threshold_call:
            regression_threshold = self.regression_threshold_call
        if regression_threshold:
            return {"label": 'racist' if score > regression_threshold else 'non-racist', "score": score}
        else:
            return {"score": score}