import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------
# 1. Sample Numerical Dataset
# -------------------------------------------------
# Features: [size_sqft, rooms, age_years]
X = np.array([
    [800, 2, 20],
    [1200, 3, 10],
    [1500, 4, 5],
    [2000, 4, 2],
], dtype=np.float32)

# Target: house price
y = np.array([50, 75, 110, 150], dtype=np.float32)

# -------------------------------------------------
# 2. Compute Statistics (TRAIN DATA ONLY)
# -------------------------------------------------
mean = X.mean(axis=0)
std = X.std(axis=0)

print("Mean:", mean)
print("Std:", std)

# -------------------------------------------------
# 3. Numerical Transforms
# -------------------------------------------------
class LogTransform:
    def __call__(self, x):
        # log(1 + x) to avoid log(0)
        return torch.log1p(x)

class StandardScalerTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

# Create transform pipeline
transform = Compose([
    LogTransform(),
    StandardScalerTransform(
        mean=torch.tensor(mean),
        std=torch.tensor(std)
    )
])

# -------------------------------------------------
# 4. Custom Dataset
# -------------------------------------------------
class HousePriceDataset(Dataset):
    def __init__(self, features, targets, transform=None):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

# -------------------------------------------------
# 5. Dataset & DataLoader
# -------------------------------------------------
dataset = HousePriceDataset(X, y, transform=transform)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

# -------------------------------------------------
# 6. Iterate Over DataLoader
# -------------------------------------------------
if __name__ == "__main__":
    for batch_x, batch_y in loader:
        print("\nBatch Features:")
        print(batch_x)
        print("Batch Targets:")
        print(batch_y)
