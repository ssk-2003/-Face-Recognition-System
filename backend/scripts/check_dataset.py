"""
Quick script to check dataset structure and statistics
"""
import os
from pathlib import Path
from collections import defaultdict

def analyze_dataset(dataset_dir):
    """Analyze dataset structure"""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ Directory not found: {dataset_dir}")
        return
    
    print("="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    print(f"\nDataset location: {dataset_path}")
    
    # Count files and folders
    total_files = 0
    total_folders = 0
    identity_stats = defaultdict(int)
    
    # Check top-level structure
    items = list(dataset_path.iterdir())
    print(f"\nTop-level items: {len(items)}")
    
    # If it has train/test folders
    has_train = (dataset_path / "train").exists()
    has_test = (dataset_path / "test").exists()
    
    if has_train or has_test:
        print("\n✓ Found train/test split structure")
        
        if has_train:
            train_path = dataset_path / "train"
            train_identities = [d for d in train_path.iterdir() if d.is_dir()]
            print(f"\nTrain set:")
            print(f"  Identities: {len(train_identities)}")
            
            for identity_dir in train_identities[:5]:  # Show first 5
                images = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))
                identity_stats[identity_dir.name] = len(images)
                print(f"    {identity_dir.name}: {len(images)} images")
            
            if len(train_identities) > 5:
                print(f"    ... and {len(train_identities) - 5} more identities")
        
        if has_test:
            test_path = dataset_path / "test"
            test_identities = [d for d in test_path.iterdir() if d.is_dir()]
            print(f"\nTest set:")
            print(f"  Identities: {len(test_identities)}")
    
    else:
        # Check if identities are directly in root
        identity_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
        print(f"\n✓ Found {len(identity_folders)} identity folders")
        
        if len(identity_folders) > 0:
            print("\nSample identities:")
            for identity_dir in identity_folders[:10]:
                images = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))
                identity_stats[identity_dir.name] = len(images)
                total_files += len(images)
                print(f"  {identity_dir.name}: {len(images)} images")
            
            if len(identity_folders) > 10:
                print(f"  ... and {len(identity_folders) - 10} more identities")
    
    # Calculate statistics
    if identity_stats:
        total_images = sum(identity_stats.values())
        avg_images = total_images / len(identity_stats) if identity_stats else 0
        
        print(f"\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        print(f"Total identities: {len(identity_stats)}")
        print(f"Total images: {total_images}")
        print(f"Average images per identity: {avg_images:.1f}")
        print(f"Min images: {min(identity_stats.values())}")
        print(f"Max images: {max(identity_stats.values())}")
    
    # Recommendations
    print(f"\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    
    if not (has_train and has_test):
        print("\n✓ Dataset needs preparation")
        print("  Run: python scripts/prepare_dataset.py \\")
        print(f"         --input {dataset_path} \\")
        print(f"         --output data/processed \\")
        print(f"         --create-split")
    else:
        print("\n✓ Dataset is already split")
        print("  You can prepare it with:")
        print("  Run: python scripts/prepare_dataset.py \\")
        print(f"         --input {dataset_path}/train \\")
        print(f"         --output data/processed")
    
    print("\n✓ After preparation, build gallery:")
    print("  1. Start service: python main.py")
    print("  2. Add identities via API or:")
    print("     python demo.py --add-identity \"Name\" photo.jpg")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import sys
    
    dataset_dir = "data/raw_dataset"
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    
    analyze_dataset(dataset_dir)
