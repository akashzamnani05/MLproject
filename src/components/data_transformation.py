import sys
from dataclasses import dataclass
import os
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts",'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        # this function is responsible for the data transformation part, return preprocessor with scaling , encoding

        try:
            numerical_features = ['reading score','writing score']
            categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            num_pipeline = Pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('One Hot Incoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))

                ]
            )

            logging.info("Cat cols encoding and num cols scaling done")

            preprocessor = ColumnTransformer(
                [
                    ('numerical pipeline',num_pipeline,numerical_features),
                    ('categorical pipeline',cat_pipeline,categorical_features)
                ]
            )


            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data read")

            preprocessing_obj = self.get_data_transformer_obj()
            target_column_name = 'math score'  

            input_feature_train_df = train_df.drop(target_column_name,axis=1)   
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(target_column_name,axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessor obj")
            input_feature_train_arr =preprocessing_obj.fit_transform(input_feature_train_df,)
            input_feature_test_arr =preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saving preproceeing objs")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

