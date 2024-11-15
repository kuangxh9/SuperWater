class DataLoaderSubset:
    def __init__(self, dataloader, max_batches):
        """
        Wraps a DataLoader to yield a maximum number of batches.

        :param dataloader: The DataLoader to wrap.
        :param max_batches: The maximum number of batches to yield.
        """
        self.dataloader = dataloader
        self.max_batches = max_batches

    def __iter__(self):
        """
        Returns an iterator that yields batches from the DataLoader up to max_batches.
        """
        batch_count = 0
        for batch in self.dataloader:
            if batch_count >= self.max_batches:
                break
            yield batch
            batch_count += 1

    def __len__(self):
        """
        Returns the maximum number of batches this DataLoader will yield.
        """
        return min(self.max_batches, len(self.dataloader))
