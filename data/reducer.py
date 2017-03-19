import numpy as np
import pandas as pd
import json
import os
from progressbar import ProgressBar

class Reducer:

    def __init__(self):
        pass

    def load_data(self, file):
        """Load the JSON data from a file into a DataFrame.

        Args:
            file: string denoting path of file.

        Returns:
            data_frame: pandas DataFrame with contents of files in it.
        """
        data_frame = pd.read_json(file, lines=True)

        return data_frame

    def save_data(self, data_frame, new_file):
        """Save a DataFrame to a JSON file.

        Args:
            data_frame: DataFrame to be saved
            original_file: filepath of original JSON data.
        """
        data_frame.to_json(new_file, orient='records', lines=True)

    def remove_faulty_row(self, data_frame, column='body', value='[deleted]'):
        """Remove faulty rows form pandas DataFrame data_frame.

        Args:
            data_frame: the location of rows to be removed
            column: the column where faulty data is looked for (default 'body')
            value: a string or list of strings which denote a faulty row (default '[deleted]')
        """
        if type(value) == str:
            value = [value]

        for val in value:
            data_frame = data_frame.loc[data_frame[column] != val]

    def remove_extraneous_columns(self, data_frame, columns):
        """Remove columns from a data_frame which are unnecessary in place.

        Args:
            data_frame: the location of columns to be removed
            columns: a column descriptor or list of columns to be removed from the data_frame
       """
        if type(columns) == str:
            columns = [columns]
        data_frame.drop(columns, 1, inplace=True)
