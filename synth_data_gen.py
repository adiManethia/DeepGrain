import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import Voronoi, voronoi_plot_2d

def generate_voronoi_grains(widhth=800, height=800, num_seeds=100, noise=10):
    # step 1: generate random seed points with noise
    np.random.seed(42) # for reproducibility
    seeds = np.random.randint(low=0, high=[widhth, height], size=(num_seeds, 2))
    seeds = seeds.astype(np.float32)
    seeds += np.random.uniform(-noise, noise, seeds.shape) # add noise
    
    # step 2: compute Voronoi tesselation
    vor = Voronoi(seeds)
    
    # step 3: plot Voronoi diagram 
    fig, ax = plt.subplots(figsize=(8,8))
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange')
    ax.set_xlim(0, widhth), ax.set_ylim(0, height)
    fig.canvas.draw()
    
    # convert plot to numpy array --> grayscale image
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.close(fig)
    
    return img

# exmplae usage
#voronoi_img = generate_voronoi_grains(num_seeds=150)
#cv2.imwrite('voronoi_grains.png', voronoi_img)

### add scratches to the image
def add_scratches(img, num_scratches=5, thickness=1):
    img_with_scratches = img.copy()
    height, width = img.shape
    for _ in range(num_scratches):
        # random start and end points
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        x2 = np.random.randint(0, width)
        y2 = np.random.randint(0, height)
        cv2.line(img_with_scratches, (x1, y1), (x2, y2), 255, thickness) # 0 means black
    return img_with_scratches

# example usage
#img_scratches  = add_scratches(voronoi_img, num_scratches=8)

#### Add pores
def add_pores(img, num_pores=2000, min_radius=1, max_radius=5):
    img_with_pores = img.copy()
    height, width = img.shape
    for _ in range(num_pores):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        radius = np.random.randint(min_radius, max_radius)
        cv2.circle(img_with_pores, (x, y), radius, 0, -1)  # Fill with black
    return img_with_pores

# Example:
#img_pores = add_pores(img_scratches, num_pores=3000)

### Guassian noise
def add_gaussian_noise(img, mean=100, std=10):
    noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
    noisy_img = cv2.addWeighted(img, 0.7, noise, 0.3, 0)
    return noisy_img

# Example:
#noisy_img = add_gaussian_noise(img_pores)

### apply distortions - blur, erosion, dilation

def apply_distortions(img):
    # Median blur (simulate polishing artifacts)
    img = cv2.medianBlur(img, 5)
    
    # Gaussian blur
    img = cv2.GaussianBlur(img, (15, 15), 0)
    
    # Adaptive thresholding (to binarize)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 2
    )
    return img

# Example:
#distorted_img = apply_distortions(noisy_img)
#cv2.imwrite("synthetic_grain.png", distorted_img)

######### full pipeline
def generate_synthetic_grain_image():
    # 1. Generate Voronoi base
    Voronoi_base = generate_voronoi_grains(num_seeds=150, noise=15)
    
    # 2. Add defects
    img = add_scratches(Voronoi_base, num_scratches=np.random.randint(2, 10))
    img = add_pores(img, num_pores=np.random.randint(2000, 4000))
    
    # 3. Add Gaussian noise
    img = add_gaussian_noise(img)
    
    # 4. Apply distortions
    img = apply_distortions(img)
    
    # 5. Crop to central region (200:600, 200:600)
    img = img[200:600, 200:600]
    
    return img, Voronoi_base

# Generate 10 synthetic images
for i in range(10):
    synthetic_img, base = generate_synthetic_grain_image()
    cv2.imwrite(f"synthetic_data/image_{i}.png", synthetic_img)
    cv2.imwrite(f"synthetic_masks/mask_{i}.png", base[200:600, 200:600])  # Save Voronoi as ground truth