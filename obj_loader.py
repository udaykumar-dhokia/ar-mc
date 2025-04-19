import numpy as np

class OBJ:
    def __init__(self, filename):
        self.vertices = []
        self.faces = []

        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    self.vertices.append([float(i) for i in line.strip().split()[1:]])
                elif line.startswith('f '):
                    face = [int(i.split('/')[0]) - 1 for i in line.strip().split()[1:]]
                    if len(face) >= 2:
                        self.faces.append(face)

        self.vertices = np.array(self.vertices)
        self.center_and_scale()

    def center_and_scale(self):
        """Normalize and center model at origin"""
        min_vals = self.vertices.min(axis=0)
        max_vals = self.vertices.max(axis=0)
        center = (min_vals + max_vals) / 2
        size = (max_vals - min_vals).max()
        self.vertices = (self.vertices - center) / size

