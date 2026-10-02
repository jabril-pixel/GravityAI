import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from skimage.metrics import structural_similarity as ssim

G             = 6.674e-11
GRID_SIZE     = 8
N             = GRID_SIZE ** 3
CELL_SIZE     = 5.0
MAX_CELL_MASS = 50.0
NOISE_LEVEL   = 0.05
GRAVITY_SCALE = 1e9
EPOCHS        = 3000
NUM_RUNS      = 10
TRAIN_SAMPLES = 2000 # زيادة البيانات
L1_OUTPUT_LAMBDA = 0.05 # عقوبة لتصفير الخلايا الفارغة

def create_random_building(seed):
    np.random.seed(seed)
    grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE))
    num_blocks = np.random.randint(2, 5)
    for _ in range(num_blocks):
        size_x = np.random.randint(2, 5)
        size_y = np.random.randint(2, 5)
        size_z = np.random.randint(2, 6)
        start_x = np.random.randint(0, GRID_SIZE - size_x)
        start_y = np.random.randint(0, GRID_SIZE - size_y)
        start_z = np.random.randint(0, GRID_SIZE - size_z)
        density = np.random.uniform(10, 50)
        grid[start_x:start_x+size_x, start_y:start_y+size_y, start_z:start_z+size_z] = density
    return grid

def get_observation_points():
    points = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            for k in range(GRID_SIZE):
                points.append([
                    (j * CELL_SIZE) + CELL_SIZE/2,
                    (i * CELL_SIZE) + CELL_SIZE/2,
                    GRID_SIZE * CELL_SIZE + (k * CELL_SIZE) + CELL_SIZE/2
                ])
    return np.array(points)

def forward_gravity(density_grid, observation_points):
    centers = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            for k in range(GRID_SIZE):
                centers.append([(j*CELL_SIZE)+CELL_SIZE/2, (i*CELL_SIZE)+CELL_SIZE/2, (k*CELL_SIZE)+CELL_SIZE/2])
    centers = np.array(centers)
    masses = density_grid.flatten() * (CELL_SIZE ** 3)
    g_obs = np.zeros(len(observation_points))
    for idx, obs_pt in enumerate(observation_points):
        for cell_idx, cell_center in enumerate(centers):
            if masses[cell_idx] > 0:
                dx = obs_pt[0] - cell_center[0]
                dy = obs_pt[1] - cell_center[1]
                dz = obs_pt[2] - cell_center[2]
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                if r > 0:
                    g_obs[idx] += G * masses[cell_idx] * dz / (r**3)
    return g_obs * GRAVITY_SCALE

def add_noise(signal, noise_level):
    noise = np.random.normal(0, noise_level * np.max(np.abs(signal)), len(signal))
    return signal + noise

class GravityNet(nn.Module):
    def __init__(self):
        super(GravityNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(N, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, N),
            nn.Sigmoid() # المخرجات بين 0 و 1
        )
    
    def forward(self, x):
        return self.network(x)

def train_and_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    obs_points = get_observation_points()
    
    # 1. توليد بيانات تدريب أكبر
    print(f"Generating {TRAIN_SAMPLES} training buildings...")
    train_buildings = []
    train_noisy = []
    for i in range(TRAIN_SAMPLES):
        building = create_random_building(seed=i*7)
        gravity = forward_gravity(building, obs_points)
        gravity_noisy = add_noise(gravity, NOISE_LEVEL)
        train_buildings.append(building.flatten())
        train_noisy.append(gravity_noisy)
    
    X_train = torch.FloatTensor(np.array(train_noisy)).to(device)
    y_train = torch.FloatTensor(np.array(train_buildings)).to(device)
    
    # 2. تطبيع البيانات (Normalization)
    X_min, X_max = X_train.min(), X_train.max()
    X_train_norm = (X_train - X_min) / (X_max - X_min + 1e-8)
    y_train_norm = y_train / MAX_CELL_MASS # تطبيع الكثافة بين 0 و 1
    
    print(f"Training data shape: X={X_train_norm.shape}, y={y_train_norm.shape}")
    
    model = GravityNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    
    # 3. التدريب مع عقوبة L1 لتصفير الخلايا الفارغة
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_train_norm)
        
        mse_loss = criterion(outputs, y_train_norm)
        # عقوبة L1 على المخرجات لإجبار الشبكة على إخراج أصفار للخلايا الفارغة
        l1_output_loss = outputs.abs().mean() 
        loss = mse_loss + L1_OUTPUT_LAMBDA * l1_output_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}')
    
    # 4. الاختبار
    model.eval()
    ssim_scores = []
    
    with torch.no_grad():
        for run in range(NUM_RUNS):
            test_building = create_random_building(seed=5000 + run*100)
            gravity_clean = forward_gravity(test_building, obs_points)
            gravity_noisy = add_noise(gravity_clean, NOISE_LEVEL)
            
            # تطبيع المدخلات بنفس معاملات التدريب
            input_tensor = torch.FloatTensor(gravity_noisy).unsqueeze(0).to(device)
            input_norm = (input_tensor - X_min) / (X_max - X_min + 1e-8)
            
            # التنبؤ وإلغاء التطبيع
            reconstructed_norm = model(input_norm).cpu().numpy().flatten()
            reconstructed = reconstructed_norm * MAX_CELL_MASS
            reconstructed = reconstructed.reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)
            
            # تطبيع القيم لتجنب تجاوز الحدود
            reconstructed = np.clip(reconstructed, 0, MAX_CELL_MASS)
            
            data_range = test_building.max() - test_building.min()
            if data_range == 0: data_range = 1.0
            score = ssim(test_building, reconstructed, data_range=data_range)
            ssim_scores.append(score)
            print(f'Run {run+1}: SSIM = {score:.4f}')
    
    print(f'\nAverage SSIM: {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}')

if __name__ == "__main__":
    train_and_evaluate()