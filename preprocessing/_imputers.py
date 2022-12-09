from typing import Tuple, Union, List, Optional, Dict

from snowflake.snowpark import DataFrame
import snowflake.snowpark.functions as F
from snowflake.snowpark import types as T

import json
from scipy import stats

from ._utilities import _check_fitted, _generate_udf_encoder, _columns_in_dataframe

__all__ = [
    "SimpleImputer",
]

def _get_numeric_columns(df: DataFrame) -> List:
    numeric_types = [T.DecimalType, T.LongType, T.DoubleType, T.FloatType, T.IntegerType]
    numeric_cols = [c.name for c in df.schema.fields if type(c.datatype) in numeric_types]

    if len(numeric_cols) == 0:
        raise ValueError(
            "No numeric columns in the provided DataFrame"
        )
    return numeric_cols


def _fix_impute_columns(df, input_columns) -> List:
    if not input_columns:
        # Get all numeric columns
        input_columns = _get_numeric_columns(df)
    else:
        # Check if list
        if not isinstance(input_columns, list):
            scale_columns = [input_columns]

    return input_columns


def _check_output_columns(output_cols, input_columns) -> List:
    if output_cols:
        if not isinstance(output_cols, list):
            output_cols = [output_cols]
        # Check so we have the same number of output columns as input columns
        if not len(input_columns) == len(output_cols):
            raise ValueError(
                f"Need the same number of output columns as input columns  Got {len(input_columns)} "
                f"input columns and {len(input_columns)} output columns"
            )
    else:
        output_cols = input_columns

    return output_cols



class SimpleImputer:
    def __init__(self, *, strategy, 
                 input_cols: Optional[Union[List[str], str]] = None,
                 output_cols: Optional[Union[List[str], str]] = None):
        """
        Simple Imputer
        """
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.strategy = strategy

    def fit(self, df):
        """
        Compute the mean and std to be used for later scaling.

        :param df: Snowpark DataFrame to be scaled
        :return: fitted encoder
        """
        # Not using sample_weight=None for now!

        # Validate data
        impute_columns = _fix_impute_columns(df, self.input_cols)
        self.input_cols = impute_columns

        if len(impute_columns) == 0:
            raise ValueError(
                "No columns to fit, the DataFrame has no numeric columns")

        obj_const_log = []
    
        if self.strategy == 'median':
            agg_func =  F.median
        elif self.strategy == 'mean': 
            agg_func =  F.mean
        elif self.strategy == 'min':
            agg_func =  F.min
        else:
            agg_func = F.mean

        for col in impute_columns:
            obj_const_log.extend([F.lit(col), F.object_construct(F.lit(self.strategy), 
                                                                 agg_func(F.col(col))
                                                                )])

        df_fitted_values = df.select(F.object_construct(*obj_const_log))

        fitted_values = json.loads(df_fitted_values.collect()[0][0])

        self.fitted_values_ = fitted_values

        return self

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Scales input columns and adds the scaled values in output columns.

        :param df: Snowpark DataFrame to be scaled.
        :return: Snowpark DataFrame with scaled columns
        """
        # Need check if fitted
        _check_fitted(self)

        scale_columns = self.input_cols

        output_cols = _check_output_columns(self.output_cols, scale_columns)
        self.output_cols = output_cols
        fitted_values = self.fitted_values_

        trans_df = df.na.fill({col:fitted_values[col][self.strategy] for col in scale_columns})

        return trans_df

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """
        Compute the mean and std and scales the DataFrame using those values.

        :param df: Snowpark DataFrame to be scaled.
        :return: Snowpark DataFrame with scaled columns

        """
        return self.fit(df).transform(df)


    def get_udf_encoder(self):
        """
        Returns the encoder as a dictionary object to be used with the udf_transform functions.

        :return: Dictionary containing fitted values
        """
        _check_fitted(self)

        return _generate_udf_encoder(self)

