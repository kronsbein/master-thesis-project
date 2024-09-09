# taken from bellamy master repository, reach out for information on source.
import numpy as np


class RepeatedRandomSubsampleInterpolationSplits:

    def __init__(self, df, n_train, n_iter):
        self.df = df
        self.xu = np.unique(df["instance_count"].values.flatten())
        self.n_train = n_train
        self.n_iter = n_iter
        self.i_iter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i_iter == self.n_iter:
            raise StopIteration

        xtrain = np.array([])
        xtest = np.array([])
        xval = np.array([])

        if self.n_train > 0:
            xtrain = np.random.choice(self.xu, size=self.n_train, replace=False)

        if self.n_train > 1:
            xtest_range = np.arange(xtrain.min(), xtrain.max() + 1)
            xtest_range = np.intersect1d(xtest_range, self.xu)
            xtest = np.atleast_1d(np.random.choice(xtest_range))

        if self.n_train > 0:
            xval_range = np.r_[np.arange(self.xu.min(),
                                         xtrain.min() + 1),
                               np.arange(xtrain.max(),
                                         self.xu.max() + 1)]
        else:
            xval_range = np.arange(self.xu.min(), self.xu.max() + 1)
        xval_range = np.intersect1d(xval_range, self.xu)
        xval = np.atleast_1d(np.random.choice(xval_range))

        x = np.r_[xtrain, xtest, xval]

        indices = np.array([
            np.random.choice(list(self.df.loc[self.df.instance_count == xi, "gross_runtime"].index))
            for xi in x
        ])
        train_indices, test_indices, val_indices = indices[:xtrain.size], indices[xtrain.size:(
            xtrain.size + xtest.size)], indices[-xval.size:]

        train_indices, test_indices, val_indices = np.array(train_indices), np.array(
            test_indices), np.array(val_indices)

        self.i_iter = self.i_iter + 1

        return train_indices, test_indices, val_indices

    def __len__(self):
        return self.n_iter
