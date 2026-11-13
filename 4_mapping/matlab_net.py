"""Apply a MATLAB patternnet exported by 3_training/export_net.m in Python (same computation as MATLAB's net(x)).

x -> remove the constant inputs -> mapminmax: (x - xoffset) * gain + ymin -> tansig hidden layer(s) -> softmax.
Returns the peat probability (softmax row 2), i.e. the Peatland Index (PI) / 100.
A regression network (one output row, the Canadian network of 05_canada_regression_vb.py) returns its output in target
units: the output mapminmax is reversed, (y - out_ymin) / out_gain + out_xoffset (= the Canadian PI, 5-100).
"""
import numpy as np
import scipy.io as sio


class MatlabNet:
    def __init__(self, path):
        m = sio.loadmat(path, squeeze_me=False)
        self.W = [np.asarray(w, dtype='float64') for w in m['W'].ravel()]
        self.b = [np.asarray(v, dtype='float64').ravel() for v in m['b'].ravel()]
        self.tf = [str(np.asarray(t).ravel()[0]) for t in m['tf'].ravel()]
        self.keep = np.asarray(m['keep']).ravel().astype(int) - 1          # MATLAB 1-based
        self.xoffset = np.asarray(m['xoffset'], dtype='float64').ravel()
        self.gain = np.asarray(m['gain'], dtype='float64').ravel()
        self.ymin = float(np.asarray(m['ymin']).ravel()[0])
        self.n_inputs = self.W[0].shape[1] if len(self.keep) == 0 else None
        self.out = None
        if 'out_gain' in m and np.asarray(m['out_gain']).size:
            self.out = tuple(float(np.asarray(m[k]).ravel()[0]) for k in ('out_xoffset', 'out_gain', 'out_ymin'))

    def predict(self, X, batch=2_000_000):
        """X: n x n_features array (same column order as in training). Returns peat probability (n,)."""
        X = np.asarray(X)
        out = np.empty(len(X), dtype='float32')
        for s in range(0, len(X), batch):
            x = X[s:s + batch].astype('float64')
            if len(self.keep):
                x = x[:, self.keep]
            a = (x - self.xoffset) * self.gain + self.ymin
            for W, b, tf in zip(self.W, self.b, self.tf):
                z = a @ W.T + b
                if tf == 'tansig':
                    a = np.tanh(z)
                elif tf == 'softmax':
                    z = z - z.max(axis=1, keepdims=True)
                    e = np.exp(z)
                    a = e / e.sum(axis=1, keepdims=True)
                elif tf == 'purelin':
                    a = z
                elif tf == 'logsig':
                    a = 1 / (1 + np.exp(-z))
                else:
                    raise ValueError(tf)
            a = a[:, 1] if a.shape[1] > 1 else a[:, 0]
            if self.out is not None:
                a = (a - self.out[2]) / self.out[1] + self.out[0]
            out[s:s + batch] = a
        return out
