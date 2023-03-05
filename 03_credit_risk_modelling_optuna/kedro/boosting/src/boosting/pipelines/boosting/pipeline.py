"""
This is a boilerplate pipeline 'boosting'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    data_prep, 
    split,
    opt,
    train,
    prediction,
    evaluate,
    plot
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func = data_prep,
            inputs = 'input_data',
            outputs = 'data_input_model',
            name = 'data_prep'
        ),

        node(
            func = split,
            inputs = 'data_input_model',
            outputs = ['df_train', 'df_test'],
            name = 'split'
        ),

        node(
            func = opt,
            inputs = ['df_train','params:optuna_metric.wasserstein'],
            outputs = ['optuna_study', 'bestparams'],
            name = 'opt'
        ),

        node(
            func = train,
            inputs = ['df_train','bestparams'],
            outputs = 'model',
            name = 'train'
        ),

        node(
            func = prediction,
            inputs = ['df_test', 'model'],
            outputs = ['y_hat','y_test'],
            name = 'prediction'
        ),

        node(
            func = evaluate,
            inputs = ['y_hat','y_test'],
            outputs = 'scores',
            name = 'evaluate'
        ),

        node(
            func = plot,
            inputs = ['y_hat', 'y_test'],
            outputs = 'fig',
            name = 'plot'
        )

    ])
