import logging
from typing import Iterable

from allennlp.common import Tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.common.registrable import Registrable
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.dataset_readers.dataset_reader import _LazyInstances
from allennlp.data.instance import Instance

logger = logging.getLogger(__name__)


class Corpus(Registrable):
    def __init__(self):
        self.train = None
        self.valid = None
        self.test = None


class FlexibleDatasetReader(DatasetReader):
    def read(self, *args, **kwargs) -> Iterable[Instance]:
        """
        Returns an ``Iterable`` containing all the instances
        in the specified dataset.

        If ``self.lazy`` is False, this calls ``self._read()``,
        ensures that the result is a list, then returns the resulting list.

        If ``self.lazy`` is True, this returns an object whose
        ``__iter__`` method calls ``self._read()`` each iteration.
        In this case your implementation of ``_read()`` must also be lazy
        (that is, not load all instances into memory at once), otherwise
        you will get a ``ConfigurationError``.

        In either case, the returned ``Iterable`` can be iterated
        over multiple times. It's unlikely you want to override this function,
        but if you do your result should likewise be repeatedly iterable.
        """
        lazy = getattr(self, 'lazy', None)
        if lazy is None:
            logger.warning("DatasetReader.lazy is not set, "
                           "did you forget to call the superclass constructor?")

        if lazy:
            return _LazyInstances(lambda: iter(self._read(*args, **kwargs)))
        else:
            instances = self._read(*args, **kwargs)
            if not isinstance(instances, list):
                instances = [instance for instance in Tqdm.tqdm(instances)]
            if not instances:
                raise ConfigurationError(
                    f"No instances were read from the given args ({args}). "
                    f"and kwargs ({kwargs})Is the path correct?")
            return instances
