import subprocess
import math
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

class DataExtractor(object):
    def __init__(self, flip: int = 0, rotate: int = 0, pathPrefix = './pdelab_data/build/output', normalizationFactor = 0.01):
        self._flip = flip
        self._rotate = rotate
        self._pathPrefix = pathPrefix
        self._normalizationFactor = normalizationFactor
        self._read_dir_data()

    def _read_dir_data(self, numNodes = 100):
        pathPrefix = self._pathPrefix
        normalizationFactor = self._normalizationFactor
        
        directories = str(subprocess.check_output('find ' + pathPrefix + '* -type d', shell=True))

        datapoints = {}
        for directory in directories.split('\\n'):
            # Each directory contain one space-time datapoint
            splits = directory.split(pathPrefix)
            if len(splits) == 2:
                # Folder name is in 'splits[1]'
                filePathPrefix = pathPrefix + splits[1] + '/'
                datapointFnames = str(subprocess.check_output('find ' + filePathPrefix + '*.raw', shell=True))
                nodeValues = np.zeros((datapointFnames.count('.raw'), numNodes)) # Initialize nodeValues

                countf = 0
                for datapointFname in datapointFnames.split('\\n'):
                    # Each file contain the state of a 2D grid at a particular time
                    fileSplits = datapointFname.split(filePathPrefix)
                    if len(fileSplits) == 2:
                        data = np.fromfile(filePathPrefix + fileSplits[1])
                        domainSize = int(data.shape[0])
                        sqrtNumNodes = int(math.sqrt(numNodes))
                        sqrtDomainSize = int(math.sqrt(domainSize))

                        # Perform rotation and flip
                        data = data.reshape(sqrtDomainSize, sqrtDomainSize) # Make it 2D
                        data = np.rot90(data, k=self._rotate, axes=(1, 0))
                        if self._flip == 1:
                            data = np.flip(data)
                        data = data.reshape(domainSize) # Switch back to 1D

                        # Extract data for the nodes
                        diff = sqrtDomainSize / sqrtNumNodes
                        start = diff / 2
                        for i in range(sqrtNumNodes):
                            x_index = start + i * diff
                            for j in range(sqrtNumNodes):
                                y_index = start + j * diff
                                nodeValues[countf, j * sqrtNumNodes + i] = normalizationFactor \
                                    * data[math.floor(y_index) * sqrtDomainSize + math.floor(x_index)]

                        countf += 1

                datapoints['output' + splits[1]] = nodeValues
        
        # Store it in the class
        self._numNodes = numNodes
        self._datapoints = datapoints

    def _get_edges(self):
        # This function constructs 'self._edges'
        sqNumNodes = math.sqrt(self._numNodes)

        # I am pretty sure that there are better ways to find the total number of edges.
        # But this works for now.
        numEdges = 0
        for i in range(self._numNodes):
            if i // sqNumNodes == (i + 1) // sqNumNodes:
                numEdges += 1
            if i + sqNumNodes < self._numNodes:
                numEdges += 1
            if (i // sqNumNodes == (i + 1) // sqNumNodes) and (i + sqNumNodes < self._numNodes):
                numEdges += 1

        self._edges = np.zeros((2, numEdges))
        count = 0
        for i in range(self._numNodes):
            if i // sqNumNodes == (i + 1) // sqNumNodes:
                self._edges[0, i] = i
                self._edges[1, i] = i + 1
                count += 1
            if i + sqNumNodes < self._numNodes:
                self._edges[0, i] = i
                self._edges[1, i] = i + sqNumNodes
                count += 1
            if (i // sqNumNodes == (i + 1) // sqNumNodes) and (i + sqNumNodes < self._numNodes):
                self._edges[0, i] = i
                self._edges[1, i] = i + sqNumNodes + 1
                count += 1


    def _get_edge_weights(self):
        self._edge_weights = np.ones(self._edges.shape[1])

    def _get_targets_and_features(self):
        self.features = []
        self.targets = []
        for stacked_target in self._datapoints.items():
            tgt = stacked_target[1]
            features = [
                tgt[i : i + self.input_lags, :].T
                for i in range(tgt.shape[0] - self.input_lags - self.output_lags)
            ]
            # Uncomment the following for temporal bundling
            '''
            targets = [
                tgt[i + self.input_lags : i + self.input_lags + self.output_lags, :].T
                for i in range(tgt.shape[0] - self.input_lags - self.output_lags)
            ]
            '''
            targets = [
                tgt[i + self.input_lags + self.output_lags, :].T
                for i in range(tgt.shape[0] - self.input_lags - self.output_lags)
            ]
            self.features.extend(features)
            self.targets.extend(targets)

    def get_dataset(self, input_lags: int = 4, output_lags: int = 1) -> StaticGraphTemporalSignal:
        """
        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)*
        """
        self.input_lags = input_lags
        self.output_lags = output_lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset

def get_empty_dataset():
    return StaticGraphTemporalSignal([], [], [], [])

def merge_datasets(dataset1, dataset2):
    return StaticGraphTemporalSignal( \
        np.concatenate((dataset1.edge_index, dataset2.edge_index), axis=1), \
        np.concatenate((dataset1.edge_weight, dataset2.edge_weight)), \
        np.concatenate((dataset1.features, dataset2.features)), \
        np.concatenate((dataset1.targets, dataset2.targets))
    )
