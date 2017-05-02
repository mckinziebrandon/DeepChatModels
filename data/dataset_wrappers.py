"""Named data wrapper classes. No added functionality to dataset base class for now,
but preprocessing checks will be incorporated into each when it's time.
"""

import logging
import os

from data._dataset import Dataset


def check_data(abs_path, name):
    """All dataset wrappers call this as a quick sanity check."""

    if abs_path is None:
        raise ValueError('No data directory found in dataset_wrappers.check_data.'
                         'Either specify data_dir or use io_utils.parse_config.')

    if os.path.basename(abs_path) != name:
        print("Data directory %s does not match dataset name %s." % (abs_path, name))
        propose_path = os.path.join(os.path.dirname(abs_path), name.lower())
        print("Would you like me to change data_dir to {}? [y/n] ".format(propose_path))
        answer = input()
        if answer == 'y':
            return propose_path
        else:
            raise ValueError("Rejected path change. Terminating program.")
    return abs_path


class Cornell(Dataset):
    """Movie dialogs."""

    def __init__(self, dataset_params):
        self._name = "cornell"
        self.log = logging.getLogger('CornellLogger')
        dataset_params['data_dir'] = check_data(
            dataset_params.get('data_dir'),
            self.name)
        super(Cornell, self).__init__(dataset_params)


class Ubuntu(Dataset):
    """Technical support chat logs from IRC."""

    def __init__(self, dataset_params):
        self._name = "ubuntu"
        self.log = logging.getLogger('UbuntuLogger')
        dataset_params['data_dir'] = check_data(
            dataset_params.get('data_dir'),
            self.name)
        super(Ubuntu, self).__init__(dataset_params)


class Reddit(Dataset):
    """Reddit comments from 2007-2015."""

    def __init__(self, dataset_params):
        self._name = "reddit"
        self.log = logging.getLogger('RedditLogger')
        dataset_params['data_dir'] = check_data(
            dataset_params.get('data_dir'),
            self.name)
        super(Reddit, self).__init__(dataset_params)


class TestData(Dataset):
    """Mock dataset with a handful of sentences."""

    def __init__(self, dataset_params):
        self.log = logging.getLogger('TestDataLogger')
        self._name = "test_data"
        dataset_params['data_dir'] = check_data(
            dataset_params.get('data_dir'),
            self.name)
        super(TestData, self).__init__(dataset_params)
