

import os
import pandas as pd
from pathlib import Path

def create_labels_from_directory(data_dir, split_name):
    """
    Create labels CSV from directory structure
    
    Args:
        data_dir: Path like 'data/test'
        split_name: 'train', 'val', or 'test'
    """
    print(f"\n📁 Processing {split_name} directory...")
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"   ⚠️  Directory not found: {data_dir}")
        return None
    
    slide_ids = []
    labels = []
    
    # Get class directories
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    if len(class_dirs) == 0:
        print(f"   ⚠️  No class directories found")
        return None
    
    print(f"   Found {len(class_dirs)} classes:")
    for class_dir in sorted(class_dirs):
        print(f"      • {class_dir.name}")
    
    # Create mapping (sorted alphabetically for consistency)
    class_to_label = {class_dir.name: idx for idx, class_dir in enumerate(sorted(class_dirs))}
    print(f"\n   Label mapping:")
    for class_name, label in class_to_label.items():
        print(f"      {class_name} → {label}")
    
    # Collect all images
    for class_dir in class_dirs:
        class_name = class_dir.name
        label = class_to_label[class_name]
        
        # Find all image files
        # Find all .npy files
        image_files = []
        for ext in ['*.npy']:  # <-- This is the only change
            image_files.extend(class_dir.glob(ext))
        
        print(f"\n   {class_name}: {len(image_files)} images")
        
        for img_file in image_files:
            # Use relative path from data/classifier/
            slide_id = f"{split_name}/{class_name}/{img_file.name}"
            slide_ids.append(slide_id)
            labels.append(label)
    
    # Create DataFrame
    df = pd.DataFrame({
        'slide_id': slide_ids,
        'label': labels
    })
    
    print(f"\n   ✅ Total: {len(df)} images")
    return df

def main():
    print("=" * 70)
    print("🏷️  Creating Labels from Folder Structure")
    print("=" * 70)
    
    base_dir = "data/classifier"
    
    # Process each split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = f"{base_dir}/{split}"
        df = create_labels_from_directory(split_dir, split)
        
        if df is not None:
            # Save labels file
            output_file = f"data/{split}_labels.csv"
            df.to_csv(output_file, index=False)
            print(f"\n   💾 Saved: {output_file}")
            
            # Show sample
            print(f"\n   📋 Sample entries:")
            print(df.head(5).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✅ Labels Created Successfully!")
    print("=" * 70)
    
    # Show summary
    print("\n📊 Dataset Summary:")
    for split in splits:
        labels_file = f"data/{split}_labels.csv"
        if os.path.exists(labels_file):
            df = pd.read_csv(labels_file)
            print(f"\n{split.upper()}:")
            print(f"   Total: {len(df)} images")
            print(f"   Class distribution:")
            for label, count in df['label'].value_counts().sort_index().items():
                print(f"      Class {label}: {count} images")
    


if __name__ == '__main__':
    main()