import os

# Change this to the path of your main folder
PARENT_FOLDER = r"C:\Users\Jon Bergland\Documents\Skole\6_semester\Bachelor-jobbing\imagenet-256\1"

# Common image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

folder_counts = []

# Loop through all items in parent folder
for item in os.listdir(PARENT_FOLDER):
    subfolder_path = os.path.join(PARENT_FOLDER, item)

    # Only process directories
    if os.path.isdir(subfolder_path):
        count = 0

        for file in os.listdir(subfolder_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                count += 1

        folder_counts.append((item, count))

# Sort by image count descending
folder_counts.sort(key=lambda x: x[1], reverse=True)

# Show top 10
print("Top 10 folders with the most images:\n")
for folder, count in folder_counts[:12]:
    print(f"{folder}: {count} images")
