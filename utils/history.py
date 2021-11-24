import matplotlib.pyplot as plt

class History:
    def __init__(self):
        self.data = dict()

    def add_record(self, name, value):
        if name not in self.data.keys():
            self.data[name] = []
        self.data[name].append(value)

    def append_history(self, history):
        if len(self.data) == 0:
            self.data = history.data
            return
        assert set(history.data.keys()) == set(self.data.keys())
        for key in self.data.keys():
            self.data[key] += history.data[key]

    def plot(self, title=None, layout='combine'):
        if layout == 'combine':
            plt.figure(figsize=(10, 5))
            N = len(self.data.items())
            for i, (key, val) in enumerate(self.data.items()):
                plt.plot(1, N, i + 1), plt.plot(val, label=key)
            plt.title(title)
            plt.legend(loc='best')
            plt.show()
        else:
            raise NotImplementedError