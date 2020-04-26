import os
import xml.etree.ElementTree as ET
import glob
import io
import codecs

from torchtext import data
from multitranslation_dataset_example import Example


class MultiSourceTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src1', fields[0]), ('src2', fields[1]),  ('trg', fields[2])]

        src1_path, src2_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with io.open(src1_path, mode='r', encoding='utf-8') as src1_file, io.open(src2_path, mode='r', encoding='utf-8') as src2_file, io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
            for src1_line, src2_line, trg_line in zip(src1_file, src2_file, trg_file):
                src1_line, src2_line, trg_line = src1_line.strip(), src2_line.strip(), trg_line.strip()
                if src1_line != '' and src2_line != '' and trg_line != '':
                    examples.append(Example.fromlist(
                        [src1_line, src2_line, trg_line], fields))

        super(MultiSourceTranslationDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, exts, fields, path=None, root='.data',
               train='train', validation='val', test='test', **kwargs):
        """Create dataset objects for splits of a TranslationDataset.

        Arguments:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            path (str): Common prefix of the splits' file paths, or None to use
                the result of cls.download(root).
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        if path is None:
            path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), exts, fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)
