from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# Folder where your PNGs are located
# Adjust the path or pattern if needed
folder = Path(r'C:\Users\lucas\PycharmProjects\02807_Project\output\data\year_clustering_dbscan')
out = folder / 'plots'
out.mkdir(parents=True, exist_ok=True)
images = sorted(glob.glob(str(folder / "*.png")))
print(len(images))
# sort out images with pca in name
images = [img for img in images if 'pca' in img]
# remove year 2001:
images = [img for img in images if '2001' not in img]
print(len(images))

fig, axes = plt.subplots(4, 5, figsize=(10, 10))  # width × height of the whole figure

for ax, imgfile in zip(axes.flatten(), images):
    img = mpimg.imread(imgfile)
    ax.imshow(img)
    ax.axis('off')  # hide axes for cleanliness

plt.tight_layout()
# plt.savefig(out / r'yearly_raw_clusterings.png', dpi=300)
plt.savefig(out / r'yearly_raw_pca_clusterings.png', dpi=300)
plt.show()
